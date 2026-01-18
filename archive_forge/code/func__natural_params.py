from numbers import Number, Real
import torch
from torch.distributions import constraints
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
@property
def _natural_params(self):
    return (self.concentration1, self.concentration0)