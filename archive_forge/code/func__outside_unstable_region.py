import math
from numbers import Number
import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import (
from torch.nn.functional import binary_cross_entropy_with_logits
def _outside_unstable_region(self):
    return torch.max(torch.le(self.probs, self._lims[0]), torch.gt(self.probs, self._lims[1]))