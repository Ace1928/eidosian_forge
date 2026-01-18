from typing import Callable, Dict, Optional, Tuple
import torch
from torch import nn
from torch.distributions import (
class ParameterProjection(nn.Module):

    def __init__(self, in_features: int, args_dim: Dict[str, int], domain_map: Callable[..., Tuple[torch.Tensor]], **kwargs) -> None:
        super().__init__(**kwargs)
        self.args_dim = args_dim
        self.proj = nn.ModuleList([nn.Linear(in_features, dim) for dim in args_dim.values()])
        self.domain_map = domain_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        params_unbounded = [proj(x) for proj in self.proj]
        return self.domain_map(*params_unbounded)