import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import (
def _clamp_by_zero(x):
    return (x.clamp(min=0) + x - x.clamp(max=0)) / 2