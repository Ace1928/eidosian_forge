from typing import List, Optional
import importlib
import torch
import torch.nn as nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
from vllm._C import ops
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.triton_kernel.prefix_prefill import (
from vllm.utils import is_hip
def _paged_attention(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, input_metadata: InputMetadata, num_kv_heads: int, scale: float, alibi_slopes: Optional[torch.Tensor]) -> torch.Tensor:
    output = torch.empty_like(query)
    block_size = value_cache.shape[3]
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = (input_metadata.max_context_len + _PARTITION_SIZE - 1) // _PARTITION_SIZE
    use_v1 = input_metadata.max_context_len <= 8192 and (max_num_partitions == 1 or num_seqs * num_heads > 512)
    if use_v1:
        ops.paged_attention_v1(output, query, key_cache, value_cache, num_kv_heads, scale, input_metadata.block_tables, input_metadata.context_lens, block_size, input_metadata.max_context_len, alibi_slopes, input_metadata.kv_cache_dtype)
    else:
        assert _PARTITION_SIZE % block_size == 0
        tmp_output = torch.empty(size=(num_seqs, num_heads, max_num_partitions, head_size), dtype=output.dtype, device=output.device)
        exp_sums = torch.empty(size=(num_seqs, num_heads, max_num_partitions), dtype=torch.float32, device=output.device)
        max_logits = torch.empty_like(exp_sums)
        ops.paged_attention_v2(output, exp_sums, max_logits, tmp_output, query, key_cache, value_cache, num_kv_heads, scale, input_metadata.block_tables, input_metadata.context_lens, block_size, input_metadata.max_context_len, alibi_slopes, input_metadata.kv_cache_dtype)
    return output