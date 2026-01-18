import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import (
@lazy_property
def _gamma(self):
    return torch.distributions.Gamma(concentration=self.total_count, rate=torch.exp(-self.logits), validate_args=False)