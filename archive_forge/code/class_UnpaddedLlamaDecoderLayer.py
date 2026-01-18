from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.llama.configuration_llama import LlamaConfig
class UnpaddedLlamaDecoderLayer(nn.Module):

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = UnpaddedLlamaAttention(config=config)
        self.mlp = UnpaddedLlamaMLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act)
        self.input_layernorm = UnpaddedLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = UnpaddedLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, cos_sin: Tuple[torch.Tensor, torch.Tensor], nz_hidden_states: torch.Tensor, nz_position_ids: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        residual = nz_hidden_states
        nz_hidden_states = self.input_layernorm(nz_hidden_states)
        nz_hidden_states = self.self_attn(cos_sin=cos_sin, nz_hidden_states=nz_hidden_states, nz_position_ids=nz_position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        nz_hidden_states = residual + nz_hidden_states
        residual = nz_hidden_states
        nz_hidden_states = self.post_attention_layernorm(nz_hidden_states)
        nz_hidden_states = self.mlp(nz_hidden_states)
        nz_hidden_states = residual + nz_hidden_states
        return nz_hidden_states