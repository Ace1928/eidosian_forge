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
class EfficientNet_B1_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/efficientnet_b1_rwightman-bac287d4.pth', transforms=partial(ImageClassification, crop_size=240, resize_size=256, interpolation=InterpolationMode.BICUBIC), meta={**_COMMON_META_V1, 'num_params': 7794184, '_metrics': {'ImageNet-1K': {'acc@1': 78.642, 'acc@5': 94.186}}, '_ops': 0.687, '_file_size': 30.134, '_docs': 'These weights are ported from the original paper.'})
    IMAGENET1K_V2 = Weights(url='https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth', transforms=partial(ImageClassification, crop_size=240, resize_size=255, interpolation=InterpolationMode.BILINEAR), meta={**_COMMON_META_V1, 'num_params': 7794184, 'recipe': 'https://github.com/pytorch/vision/issues/3995#new-recipe-with-lr-wd-crop-tuning', '_metrics': {'ImageNet-1K': {'acc@1': 79.838, 'acc@5': 94.934}}, '_ops': 0.687, '_file_size': 30.136, '_docs': "\n                These weights improve upon the results of the original paper by using a modified version of TorchVision's\n                `new training recipe\n                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.\n            "})
    DEFAULT = IMAGENET1K_V2