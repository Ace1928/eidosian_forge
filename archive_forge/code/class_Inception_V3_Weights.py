import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class Inception_V3_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth', transforms=partial(ImageClassification, crop_size=299, resize_size=342), meta={'num_params': 27161264, 'min_size': (75, 75), 'categories': _IMAGENET_CATEGORIES, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#inception-v3', '_metrics': {'ImageNet-1K': {'acc@1': 77.294, 'acc@5': 93.45}}, '_ops': 5.713, '_file_size': 103.903, '_docs': 'These weights are ported from the original paper.'})
    DEFAULT = IMAGENET1K_V1