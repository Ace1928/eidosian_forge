from functools import partial
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import shufflenetv2
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..shufflenetv2 import (
from .utils import _fuse_modules, _replace_relu, quantize_model
class ShuffleNet_V2_X1_5_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(url='https://download.pytorch.org/models/quantized/shufflenetv2_x1_5_fbgemm-d7401f05.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'recipe': 'https://github.com/pytorch/vision/pull/5906', 'num_params': 3503624, 'unquantized': ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1, '_metrics': {'ImageNet-1K': {'acc@1': 72.052, 'acc@5': 90.7}}, '_ops': 0.296, '_file_size': 3.672})
    DEFAULT = IMAGENET1K_FBGEMM_V1