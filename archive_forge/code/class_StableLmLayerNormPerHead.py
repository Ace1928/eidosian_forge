import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_stablelm import StableLmConfig
class StableLmLayerNormPerHead(nn.Module):

    def __init__(self, dim, num_heads, eps=1e-05, bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norms = nn.ModuleList([nn.LayerNorm(dim, eps=eps, bias=bias) for _ in range(self.num_heads)])

    def forward(self, hidden_states: torch.Tensor):
        states_per_heads = torch.split(hidden_states, 1, dim=1)
        return torch.cat([norm(hidden_states) for norm, hidden_states in zip(self.norms, states_per_heads)], dim=1)