from functools import partial
from typing import Any, cast, Dict, List, Optional, Union
import torch
import torch.nn as nn
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class VGG13_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/vgg13-19584684.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 133047848, '_metrics': {'ImageNet-1K': {'acc@1': 69.928, 'acc@5': 89.246}}, '_ops': 11.308, '_file_size': 507.545})
    DEFAULT = IMAGENET1K_V1