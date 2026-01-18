import os
from dataclasses import replace
from itertools import zip_longest
from typing import Any, List, Optional, Set, Tuple, Union
import torch
from ..common import _get_storage_base, get_operator, register_operator
from .attn_bias import (
from .common import (
def _convert_input_format(inp: Inputs, supports_mqa: bool) -> Tuple[Inputs, Optional[torch.Tensor], int, Optional[torch.Tensor], int, Optional[torch.Tensor]]:
    assert inp.query.ndim in [4, 5]
    query, key, value = (inp.query, inp.key, inp.value)
    batch = query.shape[0]
    seqlen_q = query.shape[1]
    seqlen_kv = key.shape[1]
    head_dim_q = query.shape[-1]
    head_dim_v = value.shape[-1]
    attn_bias = inp.attn_bias
    if isinstance(attn_bias, BlockDiagonalMask):
        attn_bias.k_seqinfo.seqstart = attn_bias.k_seqinfo.seqstart.to(inp.query.device, non_blocking=True)
        attn_bias.q_seqinfo.seqstart = attn_bias.q_seqinfo.seqstart.to(inp.query.device, non_blocking=True)
        cu_seqlen_k = attn_bias.k_seqinfo.seqstart
        cu_seqlen_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
        seqused_k = None
    elif isinstance(attn_bias, (BlockDiagonalGappyKeysMask, BlockDiagonalPaddedKeysMask)):
        attn_bias.k_seqinfo.seqstart = attn_bias.k_seqinfo.seqstart.to(inp.query.device, non_blocking=True)
        attn_bias.q_seqinfo.seqstart = attn_bias.q_seqinfo.seqstart.to(inp.query.device, non_blocking=True)
        attn_bias.k_seqinfo.seqlen = attn_bias.k_seqinfo.seqlen.to(inp.query.device, non_blocking=True)
        cu_seqlen_k = attn_bias.k_seqinfo.seqstart
        cu_seqlen_q = attn_bias.q_seqinfo.seqstart
        max_seqlen_q = attn_bias.q_seqinfo.max_seqlen
        max_seqlen_k = attn_bias.k_seqinfo.max_seqlen
        seqused_k = attn_bias.k_seqinfo.seqlen
    else:
        cu_seqlen_k = None
        cu_seqlen_q = None
        seqused_k = None
        max_seqlen_q = inp.query.shape[1]
        max_seqlen_k = inp.key.shape[1]
    if query.ndim == 5:
        assert supports_mqa

        def fold(x):
            if x.stride(3) == 0:
                return x[:, :, :, 0]
            return x.reshape([x.shape[0], x.shape[1], -1, x.shape[4]])
        query = fold(query)
        key = fold(key)
        value = fold(value)
    if key.ndim == 4 and key.stride(2) == 0 and (value.stride(2) == 0) and supports_mqa:
        key = key[:, :, :1]
        value = value[:, :, :1]
    if cu_seqlen_k is not None:
        query = query.reshape([batch * seqlen_q, -1, head_dim_q])
        key = key.reshape([batch * seqlen_kv, -1, head_dim_q])
        value = value.reshape([batch * seqlen_kv, -1, head_dim_v])
    new_inp = replace(inp, query=query, key=key, value=value)
    return (new_inp, cu_seqlen_q, max_seqlen_q, cu_seqlen_k, max_seqlen_k, seqused_k)