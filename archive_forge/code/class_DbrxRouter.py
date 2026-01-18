import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_dbrx import DbrxConfig
class DbrxRouter(nn.Module):

    def __init__(self, hidden_size: int, moe_num_experts: int, moe_top_k: int, moe_jitter_eps: Optional[float], moe_normalize_expert_weights: Optional[float]):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_normalize_expert_weights = moe_normalize_expert_weights
        self.layer = nn.Linear(self.hidden_size, self.moe_num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        if self.training and self.moe_jitter_eps is not None:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.moe_jitter_eps, 1.0 + self.moe_jitter_eps)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        weights = self.layer(hidden_states).softmax(dim=-1, dtype=torch.float32)
        top_weights, top_experts = torch.topk(weights, self.moe_top_k, dim=-1)
        top_weights_scale = torch.norm(top_weights, p=self.moe_normalize_expert_weights, dim=-1, keepdim=True) if self.moe_normalize_expert_weights is not None else 1.0
        top_weights = top_weights / top_weights_scale
        weights = weights.to(hidden_states.dtype)
        top_weights = top_weights.to(hidden_states.dtype)
        return (weights, top_weights, top_experts)