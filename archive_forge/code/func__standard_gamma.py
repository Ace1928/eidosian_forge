from numbers import Number
import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
def _standard_gamma(concentration):
    return torch._standard_gamma(concentration)