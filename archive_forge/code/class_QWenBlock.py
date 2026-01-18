from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
class QWenBlock(nn.Module):

    def __init__(self, config: PretrainedConfig, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        rope_theta = getattr(config, 'rope_theta', 10000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        self.attn = QWenAttention(config.hidden_size, config.num_attention_heads, config.max_position_embeddings, rope_theta=rope_theta, rope_scaling=rope_scaling, linear_method=linear_method)
        self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = QWenMLP(config.hidden_size, config.intermediate_size // 2, linear_method=linear_method)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata, residual: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
        else:
            hidden_states, residual = self.ln_1(hidden_states, residual)
        hidden_states = self.attn(positions=positions, hidden_states=hidden_states, kv_cache=kv_cache, input_metadata=input_metadata)
        hidden_states, residual = self.ln_2(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return (hidden_states, residual)