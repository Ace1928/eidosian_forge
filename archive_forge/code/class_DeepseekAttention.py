from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
class DeepseekAttention(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, rope_theta: float=10000, rope_scaling: Optional[Dict[str, Any]]=None, max_position_embeddings: int=8192, linear_method: Optional[LinearMethodBase]=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** (-0.5)
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads, bias=False, linear_method=linear_method)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, linear_method=linear_method)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling)
        self.attn = PagedAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output