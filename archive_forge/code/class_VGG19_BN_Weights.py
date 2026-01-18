from functools import partial
from typing import Any, cast, Dict, List, Optional, Union
import torch
import torch.nn as nn
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class VGG19_BN_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/vgg19_bn-c79401a0.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 143678248, '_metrics': {'ImageNet-1K': {'acc@1': 74.218, 'acc@5': 91.842}}, '_ops': 19.632, '_file_size': 548.143})
    DEFAULT = IMAGENET1K_V1