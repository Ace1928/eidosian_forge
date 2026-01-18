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
class SigmoidTransform(Transform):
    """
    Transform via the mapping :math:`y = \\frac{1}{1 + \\exp(-x)}` and :math:`x = \\text{logit}(y)`.
    """
    domain = constraints.real
    codomain = constraints.unit_interval
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, SigmoidTransform)

    def _call(self, x):
        return _clipped_sigmoid(x)

    def _inverse(self, y):
        finfo = torch.finfo(y.dtype)
        y = y.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
        return y.log() - (-y).log1p()

    def log_abs_det_jacobian(self, x, y):
        return -F.softplus(-x) - F.softplus(x)