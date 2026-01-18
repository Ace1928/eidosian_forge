import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
class RegNet_Y_16GF_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 83590140, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#large-models', '_metrics': {'ImageNet-1K': {'acc@1': 80.424, 'acc@5': 95.24}}, '_ops': 15.912, '_file_size': 319.49, '_docs': 'These weights reproduce closely the results of the paper using a simple training recipe.'})
    IMAGENET1K_V2 = Weights(url='https://download.pytorch.org/models/regnet_y_16gf-3e4a00f9.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'num_params': 83590140, 'recipe': 'https://github.com/pytorch/vision/issues/3995#new-recipe', '_metrics': {'ImageNet-1K': {'acc@1': 82.886, 'acc@5': 96.328}}, '_ops': 15.912, '_file_size': 319.49, '_docs': "\n                These weights improve upon the results of the original paper by using a modified version of TorchVision's\n                `new training recipe\n                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.\n            "})
    IMAGENET1K_SWAG_E2E_V1 = Weights(url='https://download.pytorch.org/models/regnet_y_16gf_swag-43afe44d.pth', transforms=partial(ImageClassification, crop_size=384, resize_size=384, interpolation=InterpolationMode.BICUBIC), meta={**_COMMON_SWAG_META, 'num_params': 83590140, '_metrics': {'ImageNet-1K': {'acc@1': 86.012, 'acc@5': 98.054}}, '_ops': 46.735, '_file_size': 319.49, '_docs': '\n                These weights are learnt via transfer learning by end-to-end fine-tuning the original\n                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.\n            '})
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(url='https://download.pytorch.org/models/regnet_y_16gf_lc_swag-f3ec0043.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=224, interpolation=InterpolationMode.BICUBIC), meta={**_COMMON_SWAG_META, 'recipe': 'https://github.com/pytorch/vision/pull/5793', 'num_params': 83590140, '_metrics': {'ImageNet-1K': {'acc@1': 83.976, 'acc@5': 97.244}}, '_ops': 15.912, '_file_size': 319.49, '_docs': '\n                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk\n                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.\n            '})
    DEFAULT = IMAGENET1K_V2