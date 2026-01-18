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
class ResNet18_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(url='https://download.pytorch.org/models/quantized/resnet18_fbgemm_16fa66dd.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 11689512, 'unquantized': ResNet18_Weights.IMAGENET1K_V1, '_metrics': {'ImageNet-1K': {'acc@1': 69.494, 'acc@5': 88.882}}, '_ops': 1.814, '_file_size': 11.238})
    DEFAULT = IMAGENET1K_FBGEMM_V1