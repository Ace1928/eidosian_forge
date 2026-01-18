import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
class MaxVit_T_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(url='https://download.pytorch.org/models/maxvit_t-bc5ab103.pth', transforms=partial(ImageClassification, crop_size=224, resize_size=224, interpolation=InterpolationMode.BICUBIC), meta={'categories': _IMAGENET_CATEGORIES, 'num_params': 30919624, 'min_size': (224, 224), 'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#maxvit', '_metrics': {'ImageNet-1K': {'acc@1': 83.7, 'acc@5': 96.722}}, '_ops': 5.558, '_file_size': 118.769, '_docs': 'These weights reproduce closely the results of the paper using a similar training recipe.'})
    DEFAULT = IMAGENET1K_V1