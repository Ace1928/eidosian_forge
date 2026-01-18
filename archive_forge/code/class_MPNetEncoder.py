import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_mpnet import MPNetConfig
class MPNetEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.layer = nn.ModuleList([MPNetLayer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets, self.n_heads)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=False, **kwargs):
        position_bias = self.compute_position_bias(hidden_states)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], position_bias, output_attentions=output_attentions, **kwargs)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)

    def compute_position_bias(self, x, position_ids=None, num_buckets=32):
        bsz, qlen, klen = (x.size(0), x.size(1), x.size(1))
        if position_ids is not None:
            context_position = position_ids[:, :, None]
            memory_position = position_ids[:, None, :]
        else:
            context_position = torch.arange(qlen, dtype=torch.long)[:, None]
            memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self.relative_position_bucket(relative_position, num_buckets=num_buckets)
        rp_bucket = rp_bucket.to(x.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        values = values.expand((bsz, -1, qlen, klen)).contiguous()
        return values

    @staticmethod
    def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret