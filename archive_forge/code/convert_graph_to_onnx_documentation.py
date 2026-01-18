import warnings
from argparse import ArgumentParser
from os import listdir, makedirs
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from packaging.version import Version, parse
from transformers.pipelines import Pipeline, pipeline
from transformers.tokenization_utils import BatchEncoding
from transformers.utils import ModelOutput, is_tf_available, is_torch_available

    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU

    Args:
        onnx_model_path: Path to location the exported ONNX model is stored

    Returns: The Path generated for the quantized
    