import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_nllb_moe import NllbMoeConfig
class NllbMoeSparseMLP(nn.Module):
    """
    Implementation of the NLLB-MoE sparse MLP module.
    """

    def __init__(self, config: NllbMoeConfig, ffn_dim: int, expert_class: nn.Module=NllbMoeDenseActDense):
        super().__init__()
        self.router = NllbMoeTop2Router(config)
        self.moe_token_dropout = config.moe_token_dropout
        self.token_dropout = nn.Dropout(self.moe_token_dropout)
        self.num_experts = config.num_experts
        self.experts = nn.ModuleDict()
        for idx in range(self.num_experts):
            self.experts[f'expert_{idx}'] = expert_class(config, ffn_dim)

    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor]=False):
        """
        The goal of this forward pass is to have the same number of operation as the equivalent `NllbMoeDenseActDense`
        (mlp) layer. This means that all of the hidden states should be processed at most twice ( since we are using a
        top_2 gating mecanism). This means that we keep the complexity to O(batch_size x sequence_length x hidden_dim)
        instead of O(num_experts x batch_size x sequence_length x hidden_dim).

        1- Get the `router_probs` from the `router`. The shape of the `router_mask` is `(batch_size X sequence_length,
        num_expert)` and corresponds to the boolean version of the `router_probs`. The inputs are masked using the
        `router_mask`.

        2- Dispatch the hidden_states to its associated experts. The router probabilities are used to weight the
        contribution of each experts when updating the masked hidden states.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_dim)`):
                The hidden states
            padding_mask (`torch.Tensor`, *optional*, defaults to `False`):
                Attention mask. Can be in the causal form or not.

        Returns:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_dim)`):
                Updated hidden states
            router_logits (`torch.Tensor` of shape `(batch_size, sequence_length, num_experts)`):
                Needed for computing the loss

        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        top_1_mask, router_probs = self.router(hidden_states, padding_mask)
        router_mask = router_probs.bool()
        hidden_states = hidden_states.reshape(batch_size * sequence_length, hidden_dim)
        masked_hidden_states = torch.einsum('bm,be->ebm', hidden_states, router_mask)
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, idx]
            combining_weights = router_probs[token_indices, idx]
            expert_output = expert(masked_hidden_states[idx, token_indices])
            if self.moe_token_dropout > 0:
                if self.training:
                    expert_output = self.token_dropout(expert_output)
                else:
                    expert_output *= 1 - self.moe_token_dropout
            masked_hidden_states[idx, token_indices] = torch.einsum('b,be->be', combining_weights, expert_output)
        hidden_states = masked_hidden_states.sum(dim=0).reshape(batch_size, sequence_length, hidden_dim)
        top_1_expert_index = torch.argmax(top_1_mask, dim=-1)
        return (hidden_states, (router_probs, top_1_expert_index))