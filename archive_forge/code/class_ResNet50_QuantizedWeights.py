from functools import partial
from typing import Any, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import (
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from .utils import _fuse_modules, _replace_relu, quantize_model
class ResNet50_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(url='https://download.pytorch.org/models/quantized/resnet50_fbgemm_bf931d71.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 25557032, 'unquantized': ResNet50_Weights.IMAGENET1K_V1, '_metrics': {'ImageNet-1K': {'acc@1': 75.92, 'acc@5': 92.814}}, '_ops': 4.089, '_file_size': 24.759})
    IMAGENET1K_FBGEMM_V2 = Weights(url='https://download.pytorch.org/models/quantized/resnet50_fbgemm-23753f79.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'num_params': 25557032, 'unquantized': ResNet50_Weights.IMAGENET1K_V2, '_metrics': {'ImageNet-1K': {'acc@1': 80.282, 'acc@5': 94.976}}, '_ops': 4.089, '_file_size': 24.953})
    DEFAULT = IMAGENET1K_FBGEMM_V2