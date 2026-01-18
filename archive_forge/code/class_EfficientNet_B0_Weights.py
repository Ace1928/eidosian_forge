import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
class EfficientNet_B0_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=256, interpolation=InterpolationMode.BICUBIC), meta={**_COMMON_META_V1, 'num_params': 5288548, '_metrics': {'ImageNet-1K': {'acc@1': 77.692, 'acc@5': 93.532}}, '_ops': 0.386, '_file_size': 20.451, '_docs': 'These weights are ported from the original paper.'})
    DEFAULT = IMAGENET1K_V1