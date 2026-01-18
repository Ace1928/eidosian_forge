from typing import Dict
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.independent import Independent
from torch.distributions.transforms import ComposeTransform, Transform
from torch.distributions.utils import _sum_rightmost
def _monotonize_cdf(self, value):
    """
        This conditionally flips ``value -> 1-value`` to ensure :meth:`cdf` is
        monotone increasing.
        """
    sign = 1
    for transform in self.transforms:
        sign = sign * transform.sign
    if isinstance(sign, int) and sign == 1:
        return value
    return sign * (value - 0.5) + 0.5