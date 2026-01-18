from typing import Callable, Dict, Optional, Tuple
import torch
from torch import nn
from torch.distributions import (
def _base_distribution(self, distr_args) -> Distribution:
    total_count, logits = distr_args
    if self.dim == 1:
        return self.distribution_class(total_count=total_count, logits=logits)
    else:
        return Independent(self.distribution_class(total_count=total_count, logits=logits), 1)