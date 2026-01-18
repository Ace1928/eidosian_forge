import re
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class DenseNet161_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/densenet161-8d451a50.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 28681000, '_metrics': {'ImageNet-1K': {'acc@1': 77.138, 'acc@5': 93.56}}, '_ops': 7.728, '_file_size': 110.369})
    DEFAULT = IMAGENET1K_V1