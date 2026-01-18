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
class ShuffleNet_V2_X2_0_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'recipe': 'https://github.com/pytorch/vision/pull/5906', 'num_params': 7393996, '_metrics': {'ImageNet-1K': {'acc@1': 76.23, 'acc@5': 93.006}}, '_ops': 0.583, '_file_size': 28.433, '_docs': "\n                These weights were trained from scratch by using TorchVision's `new training recipe\n                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.\n            "})
    DEFAULT = IMAGENET1K_V1