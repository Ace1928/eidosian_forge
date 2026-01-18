import warnings
from functools import partial
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..googlenet import BasicConv2d, GoogLeNet, GoogLeNet_Weights, GoogLeNetOutputs, Inception, InceptionAux
from .utils import _fuse_modules, _replace_relu, quantize_model
class GoogLeNet_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(url='https://download.pytorch.org/models/quantized/googlenet_fbgemm-c81f6644.pth', transforms=partial(ImageClassification, crop_size=224), meta={'num_params': 6624904, 'min_size': (15, 15), 'categories': _IMAGENET_CATEGORIES, 'backend': 'fbgemm', 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#post-training-quantized-models', 'unquantized': GoogLeNet_Weights.IMAGENET1K_V1, '_metrics': {'ImageNet-1K': {'acc@1': 69.826, 'acc@5': 89.404}}, '_ops': 1.498, '_file_size': 12.618, '_docs': '\n                These weights were produced by doing Post Training Quantization (eager mode) on top of the unquantized\n                weights listed below.\n            '})
    DEFAULT = IMAGENET1K_FBGEMM_V1