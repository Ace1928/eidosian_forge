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
class PositiveDefiniteTransform(Transform):
    """
    Transform from unconstrained matrices to positive-definite matrices.
    """
    domain = constraints.independent(constraints.real, 2)
    codomain = constraints.positive_definite

    def __eq__(self, other):
        return isinstance(other, PositiveDefiniteTransform)

    def _call(self, x):
        x = LowerCholeskyTransform()(x)
        return x @ x.mT

    def _inverse(self, y):
        y = torch.linalg.cholesky(y)
        return LowerCholeskyTransform().inv(y)