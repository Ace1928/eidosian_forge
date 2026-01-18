import functools
import math
import numbers
import operator
import weakref
from typing import List
import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.utils import (
from torch.nn.functional import pad, softplus
class SoftplusTransform(Transform):
    """
    Transform via the mapping :math:`\\text{Softplus}(x) = \\log(1 + \\exp(x))`.
    The implementation reverts to the linear function when :math:`x > 20`.
    """
    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, SoftplusTransform)

    def _call(self, x):
        return softplus(x)

    def _inverse(self, y):
        return (-y).expm1().neg().log() + y

    def log_abs_det_jacobian(self, x, y):
        return -softplus(-x)