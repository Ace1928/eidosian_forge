import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
from ..ops.misc import Conv2dNormActivation, SqueezeExcitation
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface
class RegNet_X_400MF_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth', transforms=partial(ImageClassification, crop_size=224), meta={**_COMMON_META, 'num_params': 5495976, 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#small-models', '_metrics': {'ImageNet-1K': {'acc@1': 72.834, 'acc@5': 90.95}}, '_ops': 0.414, '_file_size': 21.258, '_docs': 'These weights reproduce closely the results of the paper using a simple training recipe.'})
    IMAGENET1K_V2 = Weights(url='https://download.pytorch.org/models/regnet_x_400mf-62229a5f.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'num_params': 5495976, 'recipe': 'https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres', '_metrics': {'ImageNet-1K': {'acc@1': 74.864, 'acc@5': 92.322}}, '_ops': 0.414, '_file_size': 21.257, '_docs': "\n                These weights improve upon the results of the original paper by using a modified version of TorchVision's\n                `new training recipe\n                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.\n            "})
    DEFAULT = IMAGENET1K_V2