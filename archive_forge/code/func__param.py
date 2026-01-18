import torch
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution
@property
def _param(self):
    return self._categorical._param