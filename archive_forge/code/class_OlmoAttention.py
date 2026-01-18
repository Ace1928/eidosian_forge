from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (
from vllm.sequence import SamplerOutput
from hf_olmo import OLMoConfig
class OlmoAttention(nn.Module):
    """
    This is the attention block where the output is computed as ``Attention(LN(x))`` in ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, config: OLMoConfig, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        assert config.d_model % config.n_heads == 0
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = self.config.n_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // self.total_num_heads
        self.attn_norm = nn.LayerNorm(config.d_model, elementwise_affine=False, bias=False)
        self.att_proj = QKVParallelLinear(config.d_model, self.head_dim, self.total_num_heads, bias=config.include_bias, linear_method=linear_method)
        if self.config.rope:
            rope_theta = getattr(config, 'rope_theta', 10000)
            max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)
            self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=rope_theta)
        self.scaling = self.head_dim ** (-0.5)
        self.attn = PagedAttention(self.num_heads, self.head_dim, scale=self.scaling)
        self.attn_out = RowParallelLinear(config.d_model, config.d_model, bias=config.include_bias, linear_method=linear_method)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata) -> torch.Tensor:
        hidden_states = self.attn_norm(hidden_states)
        qkv, _ = self.att_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        if self.config.rope:
            q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.attn_out(attn_output)
        return output