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
def _get_conv_output_shape(input_size: Tuple[int, int], kernel_size: int, stride: int, padding: int) -> Tuple[int, int]:
    return ((input_size[0] - kernel_size + 2 * padding) // stride + 1, (input_size[1] - kernel_size + 2 * padding) // stride + 1)