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
class PowerTransform(Transform):
    """
    Transform via the mapping :math:`y = x^{\\text{exponent}}`.
    """
    domain = constraints.positive
    codomain = constraints.positive
    bijective = True

    def __init__(self, exponent, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.exponent, = broadcast_all(exponent)

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return PowerTransform(self.exponent, cache_size=cache_size)

    @lazy_property
    def sign(self):
        return self.exponent.sign()

    def __eq__(self, other):
        if not isinstance(other, PowerTransform):
            return False
        return self.exponent.eq(other.exponent).all().item()

    def _call(self, x):
        return x.pow(self.exponent)

    def _inverse(self, y):
        return y.pow(1 / self.exponent)

    def log_abs_det_jacobian(self, x, y):
        return (self.exponent * y / x).abs().log()

    def forward_shape(self, shape):
        return torch.broadcast_shapes(shape, getattr(self.exponent, 'shape', ()))

    def inverse_shape(self, shape):
        return torch.broadcast_shapes(shape, getattr(self.exponent, 'shape', ()))