from typing import Callable, Dict, Optional, Tuple
import torch
from torch import nn
from torch.distributions import (
class NegativeBinomialOutput(DistributionOutput):
    """
    Negative Binomial distribution output class.
    """
    args_dim: Dict[str, int] = {'total_count': 1, 'logits': 1}
    distribution_class: type = NegativeBinomial

    @classmethod
    def domain_map(cls, total_count: torch.Tensor, logits: torch.Tensor):
        total_count = cls.squareplus(total_count)
        return (total_count.squeeze(-1), logits.squeeze(-1))

    def _base_distribution(self, distr_args) -> Distribution:
        total_count, logits = distr_args
        if self.dim == 1:
            return self.distribution_class(total_count=total_count, logits=logits)
        else:
            return Independent(self.distribution_class(total_count=total_count, logits=logits), 1)

    def distribution(self, distr_args, loc: Optional[torch.Tensor]=None, scale: Optional[torch.Tensor]=None) -> Distribution:
        total_count, logits = distr_args
        if scale is not None:
            logits += scale.log()
        return self._base_distribution((total_count, logits))