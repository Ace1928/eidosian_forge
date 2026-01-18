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
class ConvNeXt_Base_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/convnext_base-6075fbad.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=232), meta={**_COMMON_META, 'num_params': 88591464, '_metrics': {'ImageNet-1K': {'acc@1': 84.062, 'acc@5': 96.87}}, '_ops': 15.355, '_file_size': 338.064})
    DEFAULT = IMAGENET1K_V1