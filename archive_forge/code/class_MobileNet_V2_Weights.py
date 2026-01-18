from functools import partial
from typing import Any, Callable, List, Optional
import torch
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
class MobileNet_V2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/mobilenet_v2-b0353104.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv2', '_metrics': {'ImageNet-1K': {'acc@1': 71.878, 'acc@5': 90.286}}, '_ops': 0.301, '_file_size': 13.555, '_docs': 'These weights reproduce closely the results of the paper using a simple training recipe.'})
    IMAGENET1K_V2 = Weights(url='https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'recipe': 'https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning', '_metrics': {'ImageNet-1K': {'acc@1': 72.154, 'acc@5': 90.822}}, '_ops': 0.301, '_file_size': 13.598, '_docs': "\n                These weights improve upon the results of the original paper by using a modified version of TorchVision's\n                `new training recipe\n                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.\n            "})
    DEFAULT = IMAGENET1K_V2