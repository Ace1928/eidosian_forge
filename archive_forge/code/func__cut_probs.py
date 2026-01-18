import math
from numbers import Number
import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import (
from torch.nn.functional import binary_cross_entropy_with_logits
def _cut_probs(self):
    return torch.where(self._outside_unstable_region(), self.probs, self._lims[0] * torch.ones_like(self.probs))