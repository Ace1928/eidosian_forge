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
class ShuffleNet_V2_X1_0_QuantizedWeights(WeightsEnum):
    IMAGENET1K_FBGEMM_V1 = Weights(url='https://download.pytorch.org/models/quantized/shufflenetv2_x1_fbgemm-1e62bb32.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 2278604, 'unquantized': ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1, '_metrics': {'ImageNet-1K': {'acc@1': 68.36, 'acc@5': 87.582}}, '_ops': 0.145, '_file_size': 2.334})
    DEFAULT = IMAGENET1K_FBGEMM_V1