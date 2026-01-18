import math
from typing import List, Optional, Tuple
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
class BaiChuanAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size: int, num_heads: int, position_embedding: str, rope_theta: float=10000, max_position_embeddings: int=8192, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = hidden_size // self.total_num_heads
        self.postion_embedding = position_embedding
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.W_pack = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_heads, bias=False, linear_method=linear_method)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, linear_method=linear_method)
        if self.postion_embedding == 'ALIBI':
            tp_rank = get_tensor_model_parallel_rank()
            head_start = tp_rank * self.num_heads
            head_end = (tp_rank + 1) * self.num_heads
            alibi_slopes = _get_alibi_slopes(self.total_num_heads)
            alibi_slopes = alibi_slopes[head_start:head_end].tolist()
            scaling = self.head_dim ** (-0.5)
            self.attn = PagedAttention(self.num_heads, self.head_dim, scaling, alibi_slopes=alibi_slopes)
        else:
            self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=self.max_position_embeddings, base=self.rope_theta)
            self.scaling = self.head_dim ** (-0.5)
            self.attn = PagedAttention(self.num_heads, self.head_dim, self.scaling)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata) -> torch.Tensor:
        qkv, _ = self.W_pack(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        if self.postion_embedding != 'ALIBI':
            q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output