from typing import List, Optional, Tuple
import torch
from torch import nn
from torch.nn import LayerNorm
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
from vllm.transformers_utils.configs import ChatGLMConfig
class GLMAttention(nn.Module):

    def __init__(self, config, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.multi_query_attention = config.multi_query_attention
        self.total_num_kv_heads = config.multi_query_group_num if config.multi_query_attention else config.num_attention_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** (-0.5)
        self.query_key_value = QKVParallelLinear(self.hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=config.add_bias_linear or config.add_qkv_bias, linear_method=linear_method)
        self.dense = RowParallelLinear(self.total_num_heads * self.head_dim, config.hidden_size, bias=config.add_bias_linear, linear_method=linear_method)
        rope_ratio = getattr(config, 'rope_ratio', 1.0)
        max_positions = getattr(config, 'seq_length', 8192)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim // 2, max_position=max_positions, base=10000 * rope_ratio, is_neox_style=False)
        self.attn = PagedAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        key_cache, value_cache = kv_cache
        context_layer = self.attn(q, k, v, key_cache, value_cache, input_metadata)
        attn_output, _ = self.dense(context_layer)
        return attn_output