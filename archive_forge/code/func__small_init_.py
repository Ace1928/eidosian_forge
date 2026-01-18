import logging
import math
from enum import Enum
from typing import Callable
import torch
import torch.nn as nn
from torch.nn.init import (
def _small_init_(tensor: torch.Tensor, gain: float=1.0) -> torch.Tensor:
    """Fills the input `Tensor` with values according to the method
    described in `Transformer Without Tears`_, using a uniform distribution.

    This is a variation of the Xavier init. The resulting tensor will have values sampled from
    :math:`\\mathcal{U}(-a, a)` where

    .. math::
        a = \\text{gain} \\times \\sqrt{\\frac{6}{\\text{fan\\_in} + 4 * \\text{fan\\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    .. _`Transformer Without Tears`: https://arxiv.org/abs/1910.05895

    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    a = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -a, a)