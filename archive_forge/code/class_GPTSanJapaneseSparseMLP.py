import copy
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from ...activations import ACT2FN
from ...modeling_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_gptsan_japanese import GPTSanJapaneseConfig
class GPTSanJapaneseSparseMLP(nn.Module):
    """
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: GPTSanJapaneseConfig, expert_class: nn.Module=GPTSanJapaneseDenseActDense):
        super().__init__()
        self.router = GPTSanJapaneseTop1Router(config)
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f'expert_{idx}'] = expert_class(config)

    def forward(self, hidden_states):
        """
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)
        next_states = hidden_states.clone()
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert(hidden_states[token_indices]).to(next_states.dtype)
        hidden_states = router_probs * next_states
        return (hidden_states, (router_logits, expert_index))