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
def _make_block_input_shapes(input_size: Tuple[int, int], n_blocks: int) -> List[Tuple[int, int]]:
    """Util function to check that the input size is correct for a MaxVit configuration."""
    shapes = []
    block_input_shape = _get_conv_output_shape(input_size, 3, 2, 1)
    for _ in range(n_blocks):
        block_input_shape = _get_conv_output_shape(block_input_shape, 3, 2, 1)
        shapes.append(block_input_shape)
    return shapes