import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
class InputProjection(nn.Module):
    """
    Handle all the input projections in one go, opportunistically fuse some operations.
    """

    def __init__(self, query_proj_params: InputProjectionConfig, key_proj_params: Optional[InputProjectionConfig], value_proj_params: Optional[InputProjectionConfig], use_separate_proj_weight: bool=True):
        super().__init__()
        self.out_features = query_proj_params.out_features
        self.q_proj = nn.Linear(query_proj_params.in_features, query_proj_params.out_features, query_proj_params.bias)
        if key_proj_params is not None:
            self.k_proj = nn.Linear(key_proj_params.in_features, key_proj_params.out_features, key_proj_params.bias)
        else:
            logger.info('No Key projection parameters were passed, assuming that the weights' + ' are shared with the query projection')
            self.k_proj = self.q_proj
        if value_proj_params is not None:
            self.v_proj = nn.Linear(value_proj_params.in_features, value_proj_params.out_features, value_proj_params.bias)
        else:
            logger.info('No Value projection parameters were passed, assuming that the weights' + ' are shared with the query projection')
            self.v_proj = self.q_proj
        if not use_separate_proj_weight:
            with torch.no_grad():
                self.k_proj.weight = self.q_proj.weight
                self.v_proj.weight = self.q_proj.weight

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = map(lambda fn, x: fn(x), [self.q_proj, self.k_proj, self.v_proj], [query, key, value])
        return (q, k, v)