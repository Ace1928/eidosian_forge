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
class ResNeXt101_64X4D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'num_params': 83455272, 'recipe': 'https://github.com/pytorch/vision/pull/5935', '_metrics': {'ImageNet-1K': {'acc@1': 83.246, 'acc@5': 96.454}}, '_ops': 15.46, '_file_size': 319.318, '_docs': "\n                These weights were trained from scratch by using TorchVision's `new training recipe\n                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.\n            "})
    DEFAULT = IMAGENET1K_V1