import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Type, Union
import torch
from ..._cpp_lib import _built_with_cuda
from ..common import BaseOperator
from .attn_bias import (
class AttentionOpBase(BaseOperator):
    """Base class for any attention operator in xFormers

    See:

    - :attr:`xformers.ops.fmha.cutlass.FwOp`
    - :attr:`xformers.ops.fmha.cutlass.BwOp`
    - :attr:`xformers.ops.fmha.flash.FwOp`
    - :attr:`xformers.ops.fmha.flash.BwOp`
    - :attr:`xformers.ops.fmha.triton.FwOp`
    - :attr:`xformers.ops.fmha.triton.BwOp`
    - :attr:`xformers.ops.fmha.small_k.FwOp`
    - :attr:`xformers.ops.fmha.small_k.BwOp`
    """
    OPERATOR: Any
    SUPPORTED_DEVICES: Set[str]
    CUDA_MINIMUM_COMPUTE_CAPABILITY: Tuple[int, int] = (5, 0)
    SUPPORTED_DTYPES: Set[torch.dtype]
    SUPPORTED_MAX_K: float
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None)}
    SUPPORTS_DROPOUT: bool
    SUPPORTS_CUSTOM_SCALE: bool = False
    SUPPORTS_DIFFERENT_VALUE_EMBED: bool = False
    SUPPORTS_OUTPUT_DTYPE: bool = False
    SUPPORTS_PARTIAL: bool = False
    IS_DETERMINISTIC: bool = True
    SUPPORTS_BMGHK: bool = False
    NAME: str
    OPERATOR_CATEGORY = 'memory_efficient_attention'
    _TEST_BATCH_SIZES: List[int] = [1, 300]
    _TEST_K: List[int] = [32, 128]

    @classmethod
    def supports(cls, d: Inputs) -> bool:
        return not cls.not_supported_reasons(d)

    @classmethod
    def shape_not_supported_reasons(cls, Mq: int, Mkv: int, K: int, Kv: int) -> List[str]:
        reasons = []
        if not cls.SUPPORTS_DIFFERENT_VALUE_EMBED and K != Kv:
            reasons.append('query.shape[-1] != value.shape[-1]')
        if max(K, Kv) > cls.SUPPORTED_MAX_K:
            reasons.append(f'max(query.shape[-1] != value.shape[-1]) > {cls.SUPPORTED_MAX_K}')
        return reasons

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        """
        Returns a list of reasons why this is not supported.
        The kernel can run these inputs only if the returned list is empty
        """
        query_shape = d.query.shape
        reasons = cls.shape_not_supported_reasons(Mq=query_shape[1], Mkv=d.key.shape[1], K=query_shape[-1], Kv=query_shape[-1] if d.value.dtype == torch.int32 else d.value.shape[-1])
        device_type = d.query.device.type
        dtype = d.query.dtype
        if device_type not in cls.SUPPORTED_DEVICES:
            reasons.append(f'device={device_type} (supported: {cls.SUPPORTED_DEVICES})')
        if device_type == 'cuda' and (not _built_with_cuda) and (torch.version.hip is None):
            reasons.append("xFormers wasn't build with CUDA support")
        if device_type == 'cuda':
            device_capability = torch.cuda.get_device_capability(d.device)
            if device_capability < cls.CUDA_MINIMUM_COMPUTE_CAPABILITY:
                reasons.append(f'requires device with capability > {cls.CUDA_MINIMUM_COMPUTE_CAPABILITY} but your GPU has capability {device_capability} (too old)')
        if dtype not in cls.SUPPORTED_DTYPES:
            reasons.append(f'dtype={dtype} (supported: {cls.SUPPORTED_DTYPES})')
        if type(d.attn_bias) not in cls.SUPPORTED_ATTN_BIAS_TYPES:
            reasons.append(f'attn_bias type is {type(d.attn_bias)}')
        if not cls.SUPPORTS_OUTPUT_DTYPE:
            if d.output_dtype is not None and d.output_dtype is not dtype:
                reasons.append('Custom output dtype not supported')
        if d.is_partial and (not cls.SUPPORTS_PARTIAL):
            reasons.append('Partial attention not supported')
        if d.p != 0.0 and (not cls.SUPPORTS_DROPOUT):
            reasons.append('dropout > 0.0')
        if d.scale is not None and (not cls.SUPPORTS_CUSTOM_SCALE):
            reasons.append('has custom scale')
        if dtype is torch.bfloat16 and (not device_type.startswith('cuda') or torch.cuda.get_device_capability(d.query.device)[0] < 8):
            reasons.append('bf16 is only supported on A100+ GPUs')
        if not cls.is_available():
            reasons.append("operator wasn't built - see `python -m xformers.info` for more info")
        if not cls.IS_DETERMINISTIC and torch.are_deterministic_algorithms_enabled():
            reasons.append('operator is non-deterministic, but `torch.use_deterministic_algorithms` is set')
        if not cls.SUPPORTS_BMGHK and d.query.ndim == 5:
            reasons.append('operator does not support BMGHK format')
        return reasons