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
class IndependentTransform(Transform):
    """
    Wrapper around another transform to treat
    ``reinterpreted_batch_ndims``-many extra of the right most dimensions as
    dependent. This has no effect on the forward or backward transforms, but
    does sum out ``reinterpreted_batch_ndims``-many of the rightmost dimensions
    in :meth:`log_abs_det_jacobian`.

    Args:
        base_transform (:class:`Transform`): A base transform.
        reinterpreted_batch_ndims (int): The number of extra rightmost
            dimensions to treat as dependent.
    """

    def __init__(self, base_transform, reinterpreted_batch_ndims, cache_size=0):
        super().__init__(cache_size=cache_size)
        self.base_transform = base_transform.with_cache(cache_size)
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return IndependentTransform(self.base_transform, self.reinterpreted_batch_ndims, cache_size=cache_size)

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        return constraints.independent(self.base_transform.domain, self.reinterpreted_batch_ndims)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return constraints.independent(self.base_transform.codomain, self.reinterpreted_batch_ndims)

    @property
    def bijective(self):
        return self.base_transform.bijective

    @property
    def sign(self):
        return self.base_transform.sign

    def _call(self, x):
        if x.dim() < self.domain.event_dim:
            raise ValueError('Too few dimensions on input')
        return self.base_transform(x)

    def _inverse(self, y):
        if y.dim() < self.codomain.event_dim:
            raise ValueError('Too few dimensions on input')
        return self.base_transform.inv(y)

    def log_abs_det_jacobian(self, x, y):
        result = self.base_transform.log_abs_det_jacobian(x, y)
        result = _sum_rightmost(result, self.reinterpreted_batch_ndims)
        return result

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.base_transform)}, {self.reinterpreted_batch_ndims})'

    def forward_shape(self, shape):
        return self.base_transform.forward_shape(shape)

    def inverse_shape(self, shape):
        return self.base_transform.inverse_shape(shape)