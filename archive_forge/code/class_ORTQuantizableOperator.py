import os
import re
from enum import Enum
from inspect import signature
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from packaging import version
from transformers.utils import logging
import onnxruntime as ort
from ..exporters.onnx import OnnxConfig, OnnxConfigWithLoss
from ..utils.import_utils import _is_package_available
class ORTQuantizableOperator(Enum):
    Gather = 'Gather'
    Transpose = 'Transpose'
    EmbedLayerNormalizationQuant = 'EmbedLayerNormalization'
    Conv = 'Conv'
    MatMul = 'MatMul'
    Add = 'Add'
    Mul = 'Mul'
    Relu = 'Relu'
    Clip = 'Clip'
    LeakyRelu = 'LeakyRelu'
    Sigmoid = 'Sigmoid'
    MaxPool = 'MaxPool'
    GlobalAveragePool = 'GlobalAveragePool'
    Split = 'Split'
    Pad = 'Pad'
    Reshape = 'Reshape'
    Squeeze = 'Squeeze'
    Unsqueeze = 'Unsqueeze'
    Resize = 'Resize'
    AveragePool = 'AveragePool'
    Concat = 'Concat'