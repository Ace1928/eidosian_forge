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
def _make_alibi_bias(alibi_slopes: torch.Tensor, num_kv_heads: int, batch_size: int, seq_len: int, dtype: torch.dtype) -> LowerTriangularMaskWithTensorBias:
    bias = torch.arange(seq_len, dtype=dtype)
    bias = bias[None, :] - bias[:, None]
    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(batch_size, num_heads, seq_len, padded_len, device=alibi_slopes.device, dtype=dtype)[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
    attn_bias = LowerTriangularMaskWithTensorBias(bias)
    return attn_bias