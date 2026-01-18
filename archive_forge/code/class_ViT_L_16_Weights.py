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
class ViT_L_16_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/vit_l_16-852ce7e3.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=242), meta={**_COMMON_META, 'num_params': 304326632, 'min_size': (224, 224), 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#vit_l_16', '_metrics': {'ImageNet-1K': {'acc@1': 79.662, 'acc@5': 94.638}}, '_ops': 61.555, '_file_size': 1161.023, '_docs': "\n                These weights were trained from scratch by using a modified version of TorchVision's\n                `new training recipe\n                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.\n            "})
    IMAGENET1K_SWAG_E2E_V1 = Weights(url='https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth', transforms=partial(ImageClassification, crop_size=512, resize_size=512, interpolation=InterpolationMode.BICUBIC), meta={**_COMMON_SWAG_META, 'num_params': 305174504, 'min_size': (512, 512), '_metrics': {'ImageNet-1K': {'acc@1': 88.064, 'acc@5': 98.512}}, '_ops': 361.986, '_file_size': 1164.258, '_docs': '\n                These weights are learnt via transfer learning by end-to-end fine-tuning the original\n                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.\n            '})
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(url='https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=224, interpolation=InterpolationMode.BICUBIC), meta={**_COMMON_SWAG_META, 'recipe': 'https://github.com/pytorch/vision/pull/5793', 'num_params': 304326632, 'min_size': (224, 224), '_metrics': {'ImageNet-1K': {'acc@1': 85.146, 'acc@5': 97.422}}, '_ops': 61.555, '_file_size': 1161.023, '_docs': '\n                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk\n                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.\n            '})
    DEFAULT = IMAGENET1K_V1