from functools import partial
from typing import Any, Callable, List, Optional, Sequence
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ..ops.misc import Conv2dNormActivation, Permute
from ..ops.stochastic_depth import StochasticDepth
from ..transforms._presets import ImageClassification
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class ConvNeXt_Small_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/convnext_small-0c510722.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=230), meta={**_COMMON_META, 'num_params': 50223688, '_metrics': {'ImageNet-1K': {'acc@1': 83.616, 'acc@5': 96.65}}, '_ops': 8.684, '_file_size': 191.703})
    DEFAULT = IMAGENET1K_V1