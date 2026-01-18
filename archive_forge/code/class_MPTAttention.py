import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.mpt import MPTConfig
class MPTAttention(nn.Module):

    def __init__(self, config: MPTConfig, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.d_model = config.d_model
        self.total_num_heads = config.n_heads
        self.head_dim = self.d_model // self.total_num_heads
        self.clip_qkv = config.attn_config['clip_qkv']
        self.qk_ln = config.attn_config['qk_ln']
        self.alibi_bias_max = config.attn_config['alibi_bias_max']
        if 'kv_n_heads' in config.attn_config:
            self.total_num_kv_heads = config.attn_config['kv_n_heads']
        else:
            self.total_num_kv_heads = self.total_num_heads
        assert not config.attn_config['prefix_lm']
        assert config.attn_config['alibi']
        self.Wqkv = QKVParallelLinear(self.d_model, self.d_model // self.total_num_heads, self.total_num_heads, self.total_num_kv_heads, bias=not config.no_bias, linear_method=linear_method)
        if self.qk_ln:
            self.q_ln = nn.LayerNorm(self.d_model)
            self.k_ln = nn.LayerNorm(self.d_model)
        self.out_proj = RowParallelLinear(self.d_model, self.d_model, bias=not config.no_bias, linear_method=linear_method)
        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size
        if self.total_num_kv_heads >= tp_world_size:
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        tp_rank = get_tensor_model_parallel_rank()
        head_start = tp_rank * self.num_heads
        head_end = (tp_rank + 1) * self.num_heads
        alibi_slopes = _get_alibi_slopes(self.total_num_heads, self.alibi_bias_max)
        alibi_slopes = alibi_slopes[head_start:head_end].tolist()
        self.head_dim = self.d_model // self.total_num_heads
        scaling = self.head_dim ** (-0.5)
        self.attn = PagedAttention(self.num_heads, self.head_dim, scaling, alibi_slopes=alibi_slopes, num_kv_heads=self.num_kv_heads)

    def forward(self, position_ids: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata) -> torch.Tensor:
        del position_ids
        qkv, _ = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.qk_ln:
            q = self.q_ln(q)
            k = self.k_ln(k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.out_proj(attn_output)
        return output