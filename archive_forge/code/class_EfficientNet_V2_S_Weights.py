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
class EfficientNet_V2_S_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth', transforms=partial(ImageClassification, crop_size=384, resize_size=384, interpolation=InterpolationMode.BILINEAR), meta={**_COMMON_META_V2, 'num_params': 21458488, '_metrics': {'ImageNet-1K': {'acc@1': 84.228, 'acc@5': 96.878}}, '_ops': 8.366, '_file_size': 82.704, '_docs': "\n                These weights improve upon the results of the original paper by using a modified version of TorchVision's\n                `new training recipe\n                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.\n            "})
    DEFAULT = IMAGENET1K_V1