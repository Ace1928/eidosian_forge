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
class ResNeXt101_32X8D_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(url='https://download.pytorch.org/models/quantized/resnext101_32x8_fbgemm_09835ccf.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 88791336, 'unquantized': ResNeXt101_32X8D_Weights.IMAGENET1K_V1, '_metrics': {'ImageNet-1K': {'acc@1': 78.986, 'acc@5': 94.48}}, '_ops': 16.414, '_file_size': 86.034})
    IMAGENET1K_FBGEMM_V2 = Weights(url='https://download.pytorch.org/models/quantized/resnext101_32x8_fbgemm-ee16d00c.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'num_params': 88791336, 'unquantized': ResNeXt101_32X8D_Weights.IMAGENET1K_V2, '_metrics': {'ImageNet-1K': {'acc@1': 82.574, 'acc@5': 96.132}}, '_ops': 16.414, '_file_size': 86.645})
    DEFAULT = IMAGENET1K_FBGEMM_V2