from functools import partial
from typing import Any, Callable, List, Optional, Sequence
import torch
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation as SElayer
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
class MobileNet_V3_Large_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 5483032, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small', '_metrics': {'ImageNet-1K': {'acc@1': 74.042, 'acc@5': 91.34}}, '_ops': 0.217, '_file_size': 21.114, '_docs': 'These weights were trained from scratch by using a simple training recipe.'})
    IMAGENET1K_V2 = Weights(url='https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'num_params': 5483032, 'recipe': 'https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning', '_metrics': {'ImageNet-1K': {'acc@1': 75.274, 'acc@5': 92.566}}, '_ops': 0.217, '_file_size': 21.107, '_docs': "\n                These weights improve marginally upon the results of the original paper by using a modified version of\n                TorchVision's `new training recipe\n                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.\n            "})
    DEFAULT = IMAGENET1K_V2