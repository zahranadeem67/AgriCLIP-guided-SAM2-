
# AgriCLIP-Guided SAM2 for Agricultural Vegetation Segmentation

## 1. Project Overview

This project investigates semantic-guided image segmentation for agricultural vegetation using DINO ResNet-50, AgriCLIP, and Segment Anything Model 2 (SAM2).

The main objective is to reduce the need for manually provided spatial prompts by using semantic information to automatically identify vegetation-related regions and generate bounding-box prompts for SAM2.

The framework combines visual feature extraction, semantic alignment, automatic spatial prompt generation, and pixel-level vegetation segmentation.

The project also investigates different SAM2 prompt configurations, fine-tuned decoder checkpoints, and single-mask versus multi-mask prediction strategies.

## 2. Proposed Methodology

The proposed AgriCLIP-guided SAM2 pipeline consists of the following stages:

### 2.1 Visual Feature Extraction

A pretrained DINO ResNet-50 model extracts visual features from agricultural images.

The Layer 4 feature map produces a 7 × 7 spatial grid, resulting in 49 spatial descriptors, each with 2048 dimensions.

The input image for DINO feature extraction is resized to 224 × 224 pixels.

### 2.2 Semantic Alignment

A trained linear alignment module projects the 2048-dimensional DINO features into a 512-dimensional semantic embedding space aligned with CLIP/AgriCLIP text representations.

The DINO backbone and CLIP text encoder remain frozen, while the linear alignment module is trained separately.

### 2.3 Semantic Prompt Generation

Positive vegetation concepts and negative background concepts are used to calculate semantic similarity scores.

The semantic scores are used to rank the spatial descriptors, and the top 32 descriptors are selected.

Their spatial coordinates are combined to generate a single bounding-box prompt. An additional margin of 25 pixels is applied to provide spatial context.

### 2.4 SAM2 Segmentation

The generated bounding-box prompt is passed to SAM2. The model produces pixel-level vegetation segmentation masks.

Different SAM2 configurations are investigated, including single-mask and multi-mask prediction.

### 2.5 Multi-Mask Candidate Evaluation

The multi-mask experiment enables SAM2 to generate multiple candidate masks for the same bounding-box prompt.

The candidate with the highest SAM2-predicted IoU confidence score is selected.

The candidate masks are also evaluated against ground-truth masks to investigate candidate quality and the reliability of SAM2 confidence-based selection.

## 3. Datasets

The experiments involve the following agricultural vegetation datasets:

- VegAnn
- ReLeaf (CropAndWeedAndLeaf)
- WeedsGalore
- EWI1S

The datasets are not included in this repository and must be obtained separately from their respective sources.

Dataset-specific image and mask paths must be configured in the notebooks before running the experiments.

### VegAnn Dataset

The VegAnn dataset is loaded from Hugging Face:

https://huggingface.co/datasets/simonMadec/VegAnn

The multi-mask evaluation notebook uses a deterministic 80/20 training-validation split with random seed 42.

The dataset contains 3,775 images, with 3,020 training images and 755 validation images. After excluding empty validation masks, 746 validation images remain for evaluation.

## 4. Repository Structure

```text
AgriCLIP-guided-SAM2-/
│
├── notebooks/
│   └── Jupyter notebooks for data preparation,
│       feature extraction, semantic alignment,
│       model training, and evaluation
│
├── results/
│   └── CSV files containing experimental
│       results, evaluation metrics, and
│       ablation study results
│
├── visualizations/
│   └── PNG images showing segmentation
│       predictions, ground-truth masks,
│       bounding boxes, and visual comparisons
│
└── README.md
```

## 5. Technologies and Libraries

The project uses:

- Python 3.10.12
- PyTorch 2.12.0+cu130
- CUDA 13.0
- torchvision
- Hugging Face Transformers
- Hugging Face Datasets
- DINO ResNet-50
- CLIP / AgriCLIP
- Segment Anything Model 2.1
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Pillow
- tqdm
- Jupyter Notebook

## 6. Computational Environment

The recorded environment for the multi-mask evaluation experiment was:

| Component | Configuration |
|---|---|
| Python | 3.10.12 |
| PyTorch | 2.12.0+cu130 |
| GPU | NVIDIA GeForce RTX 4090 |
| CUDA | 13.0 |
| SAM2 model | facebook/sam2.1-hiera-tiny |
| DINO backbone | DINO ResNet-50 |
| DINO input resolution | 224 × 224 |
| DINO feature grid | 7 × 7 |
| Number of spatial descriptors | 49 |
| Descriptor dimension | 2048 |
| Aligned embedding dimension | 512 |
| Top-K selection | 32 |
| Bounding-box margin | 25 pixels |
| Mask threshold | 0.5 |
| Random seed | 42 |

The computational requirements depend on the model configuration, input image resolution, batch size, and dataset size.

## 7. Installation Instructions

### Step 1: Clone the repository

```bash
git clone https://github.com/zahranadeem67/AgriCLIP-guided-SAM2-.git

cd AgriCLIP-guided-SAM2-
```

### Step 2: Create a Python environment

```bash
python -m venv agriclip_env
```

Activate the environment.

**Windows:**

```bash
agriclip_env\Scripts\activate
```

**Linux:**

```bash
source agriclip_env/bin/activate
```

### Step 3: Install PyTorch

Install a PyTorch version compatible with your GPU and CUDA environment.

The recorded experimental environment used PyTorch 2.12.0 with CUDA 13.0.

### Step 4: Install additional dependencies

```bash
pip install torchvision transformers datasets
pip install numpy pandas matplotlib opencv-python pillow tqdm jupyter
```

Note: The exact versions of all dependencies were not recorded in the original notebook. The commands above provide the required libraries but may not reproduce the original environment exactly.

### Step 5: Obtain the required model checkpoints

The experiments require the following pretrained or fine-tuned components:

- DINO ResNet-50 pretrained weights.
- AgriCLIP linear alignment checkpoint.
- SAM2.1 Hiera-Tiny pretrained model weights.
- Fine-tuned SAM2 mask decoder checkpoints.

The SAM2 implementation and model documentation are available at:

https://github.com/facebookresearch/sam2

The Hugging Face SAM2 model is available at:

https://huggingface.co/facebook/sam2.1-hiera-tiny

The AgriCLIP alignment modules and corresponding checkpoints must also be made available separately.

## 8. Steps to Reproduce the Experiments

The Jupyter notebooks are provided in the `notebooks/` directory.

The general workflow is:

1. Download and prepare the agricultural datasets.
2. Configure dataset directories and checkpoint paths in the notebooks.
3. Load the pretrained DINO ResNet-50 model.
4. Load the trained AgriCLIP linear alignment module.
5. Extract spatial descriptors from DINO Layer 4.
6. Calculate semantic similarity using positive vegetation and negative background concepts.
7. Select the top-K spatial descriptors.
8. Generate automatic bounding-box prompts.
9. Load the corresponding pretrained or fine-tuned SAM2 decoder checkpoint.
10. Generate segmentation predictions.
11. Evaluate predictions against ground-truth masks.
12. Save the experimental results and visualizations.

### Multi-Mask Evaluation

The `Multimask_SAM2_36(2).ipynb` notebook investigates single-mask and multi-mask SAM2 prediction.

It evaluates the following decoder checkpoint configurations:

- VegAnn baseline
- ReLeaf fine-tuned
- WeedsGalore fine-tuned
- EWI1S fine-tuned
- Joint fine-tuned

The notebook evaluates these configurations across the agricultural datasets and compares:

- Single-mask IoU and Dice.
- Multi-mask IoU and Dice.
- Precision and recall.
- SAM2 confidence scores.
- Bounding-box IoU.
- Multi-mask candidate selection.
- Ground-truth IoU for individual candidate masks.

The notebook loads existing trained decoder checkpoints. It performs evaluation and candidate-mask analysis rather than training the SAM2 decoder.

## 9. Evaluation Metrics

The segmentation performance is evaluated using:

- Intersection over Union (IoU)
- Dice Score
- Precision
- Recall

Additional analyses include bounding-box IoU, SAM2 confidence scores, candidate-mask comparisons, and the difference between single-mask and multi-mask performance.

## 10. Reproducibility Notes and Limitations

- Agricultural datasets are not distributed with this repository.
- Large pretrained model weights and fine-tuned checkpoints are not included.
- The AgriCLIP alignment modules and checkpoints must be obtained separately.
- Dataset paths, checkpoint paths, and custom AgriCLIP module paths must be updated according to the local environment.
- The exact versions of all Python dependencies were not recorded.
- Training and inference times depend on hardware and experimental configurations.
- Results may vary depending on software versions, random seeds, dataset preparation, and model checkpoints.

For exact reproduction, use the corresponding notebook, dataset split, checkpoint, and experimental configuration.

## 11. Future Research Directions

Potential extensions of this work include:

- Adaptive semantic patch selection instead of fixed top-K selection.
- Multiple bounding-box generation for spatially distributed vegetation.
- Confidence-based automatic prompt selection.
- Improved separation of vegetation and background regions.
- Investigation of newer visual foundation models, including DINOv2 and DINOv3.
- Improving the robustness of semantic-guided segmentation across different agricultural datasets.

## Acknowledgements

This work was conducted as part of a Master's thesis on semantic understanding and segmentation for agricultural images.
