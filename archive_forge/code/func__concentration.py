import math
import torch
import torch.jit
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property
@lazy_property
def _concentration(self):
    return self.concentration.to(torch.double)