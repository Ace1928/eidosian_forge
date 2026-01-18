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
def _inv_call(self, y):
    """
        Inverts the transform `y => x`.
        """
    if self._cache_size == 0:
        return self._inverse(y)
    x_old, y_old = self._cached_x_y
    if y is y_old:
        return x_old
    x = self._inverse(y)
    self._cached_x_y = (x, y)
    return x