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
class EfficientNet_B4_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth', transforms=partial(ImageClassification, crop_size=380, resize_size=384, interpolation=InterpolationMode.BICUBIC), meta={**_COMMON_META_V1, 'num_params': 19341616, '_metrics': {'ImageNet-1K': {'acc@1': 83.384, 'acc@5': 96.594}}, '_ops': 4.394, '_file_size': 74.489, '_docs': 'These weights are ported from the original paper.'})
    DEFAULT = IMAGENET1K_V1