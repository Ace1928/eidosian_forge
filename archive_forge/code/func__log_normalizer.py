from numbers import Number, Real
import torch
from torch.distributions import constraints
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
def _log_normalizer(self, x, y):
    return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)