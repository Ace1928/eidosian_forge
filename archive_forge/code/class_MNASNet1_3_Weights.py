import warnings
from functools import partial
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class MNASNet1_3_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/mnasnet1_3-a4c69d6f.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'recipe': 'https://github.com/pytorch/vision/pull/6019', 'num_params': 6282256, '_metrics': {'ImageNet-1K': {'acc@1': 76.506, 'acc@5': 93.522}}, '_ops': 0.526, '_file_size': 24.246, '_docs': "\n                These weights were trained from scratch by using TorchVision's `new training recipe\n                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.\n            "})
    DEFAULT = IMAGENET1K_V1