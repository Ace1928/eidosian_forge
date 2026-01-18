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
class _InverseTransform(Transform):
    """
    Inverts a single :class:`Transform`.
    This class is private; please instead use the ``Transform.inv`` property.
    """

    def __init__(self, transform: Transform):
        super().__init__(cache_size=transform._cache_size)
        self._inv: Transform = transform

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        assert self._inv is not None
        return self._inv.codomain

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        assert self._inv is not None
        return self._inv.domain

    @property
    def bijective(self):
        assert self._inv is not None
        return self._inv.bijective

    @property
    def sign(self):
        assert self._inv is not None
        return self._inv.sign

    @property
    def inv(self):
        return self._inv

    def with_cache(self, cache_size=1):
        assert self._inv is not None
        return self.inv.with_cache(cache_size).inv

    def __eq__(self, other):
        if not isinstance(other, _InverseTransform):
            return False
        assert self._inv is not None
        return self._inv == other._inv

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self._inv)})'

    def __call__(self, x):
        assert self._inv is not None
        return self._inv._inv_call(x)

    def log_abs_det_jacobian(self, x, y):
        assert self._inv is not None
        return -self._inv.log_abs_det_jacobian(y, x)

    def forward_shape(self, shape):
        return self._inv.inverse_shape(shape)

    def inverse_shape(self, shape):
        return self._inv.forward_shape(shape)