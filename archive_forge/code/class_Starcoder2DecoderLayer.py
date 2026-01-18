from typing import List, Optional, Tuple
import torch
from torch import nn
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_world_size
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
class Starcoder2DecoderLayer(nn.Module):

    def __init__(self, config: Starcoder2Config, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Starcoder2Attention(config, linear_method=linear_method)
        self.mlp = Starcoder2MLP(config, linear_method=linear_method)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, kv_cache=kv_cache, input_metadata=input_metadata)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states