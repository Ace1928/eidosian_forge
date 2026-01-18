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
class SoftmaxTransform(Transform):
    """
    Transform from unconstrained space to the simplex via :math:`y = \\exp(x)` then
    normalizing.

    This is not bijective and cannot be used for HMC. However this acts mostly
    coordinate-wise (except for the final normalization), and thus is
    appropriate for coordinate-wise optimization algorithms.
    """
    domain = constraints.real_vector
    codomain = constraints.simplex

    def __eq__(self, other):
        return isinstance(other, SoftmaxTransform)

    def _call(self, x):
        logprobs = x
        probs = (logprobs - logprobs.max(-1, True)[0]).exp()
        return probs / probs.sum(-1, True)

    def _inverse(self, y):
        probs = y
        return probs.log()

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError('Too few dimensions on input')
        return shape

    def inverse_shape(self, shape):
        if len(shape) < 1:
            raise ValueError('Too few dimensions on input')
        return shape