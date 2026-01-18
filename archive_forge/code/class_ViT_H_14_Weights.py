import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
import torch
import torch.nn as nn
from ..ops.misc import Conv2dNormActivation, MLP
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class ViT_H_14_Weights(WeightsEnum):
    IMAGENET1K_SWAG_E2E_V1 = Weights(url='https://download.pytorch.org/models/vit_h_14_swag-80465313.pth', transforms=partial(ImageClassification, crop_size=518, resize_size=518, interpolation=InterpolationMode.BICUBIC), meta={**_COMMON_SWAG_META, 'num_params': 633470440, 'min_size': (518, 518), '_metrics': {'ImageNet-1K': {'acc@1': 88.552, 'acc@5': 98.694}}, '_ops': 1016.717, '_file_size': 2416.643, '_docs': '\n                These weights are learnt via transfer learning by end-to-end fine-tuning the original\n                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.\n            '})
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(url='https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=224, interpolation=InterpolationMode.BICUBIC), meta={**_COMMON_SWAG_META, 'recipe': 'https://github.com/pytorch/vision/pull/5793', 'num_params': 632045800, 'min_size': (224, 224), '_metrics': {'ImageNet-1K': {'acc@1': 85.708, 'acc@5': 97.73}}, '_ops': 167.295, '_file_size': 2411.209, '_docs': '\n                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk\n                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.\n            '})
    DEFAULT = IMAGENET1K_SWAG_E2E_V1