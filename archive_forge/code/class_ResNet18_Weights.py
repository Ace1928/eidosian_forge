from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class ResNet18_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/resnet18-f37072fd.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 11689512, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#resnet', '_metrics': {'ImageNet-1K': {'acc@1': 69.758, 'acc@5': 89.078}}, '_ops': 1.814, '_file_size': 44.661, '_docs': 'These weights reproduce closely the results of the paper using a simple training recipe.'})
    DEFAULT = IMAGENET1K_V1