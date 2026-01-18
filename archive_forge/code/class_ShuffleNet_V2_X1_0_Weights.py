from functools import partial
from typing import Any, Callable, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class ShuffleNet_V2_X1_0_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 2278604, '_metrics': {'ImageNet-1K': {'acc@1': 69.362, 'acc@5': 88.316}}, '_ops': 0.145, '_file_size': 8.791, '_docs': 'These weights were trained from scratch to reproduce closely the results of the paper.'})
    DEFAULT = IMAGENET1K_V1