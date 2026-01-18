from dataclasses import replace
from enum import Enum
from functools import partial
from typing import Any, List, Mapping, Optional, Set, Tuple, Union
import torch
from ..common import get_xformers_operator, register_operator
from . import attn_bias
from .attn_bias import (
from .common import (
@register_operator
class FwOp(AttentionFwOpBase):
    """xFormers' MHA kernel based on Composable Kernel."""
    OPERATOR = get_xformers_operator('efficient_attention_forward_ck')
    SUPPORTED_DEVICES: Set[str] = {'cuda'}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = 256
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), torch.Tensor, LowerTriangularMask, LowerTriangularFromBottomRightMask, LowerTriangularFromBottomRightLocalAttentionMask, LowerTriangularMaskWithTensorBias, BlockDiagonalMask, BlockDiagonalCausalMask, BlockDiagonalCausalWithOffsetPaddedKeysMask, attn_bias.BlockDiagonalCausalFromBottomRightMask, attn_bias.BlockDiagonalCausalLocalAttentionMask, BlockDiagonalCausalLocalAttentionFromBottomRightMask}
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = True
    SUPPORTS_BMGHK = True
    NAME = 'ckF'
    ERROR_ATOL: Mapping[torch.dtype, float] = {torch.float: 0.0003, torch.half: 0.004, torch.bfloat16: 0.028}
    ERROR_RTOL: Mapping[torch.dtype, float] = {torch.float: 2e-05, torch.half: 0.0004, torch.bfloat16: 0.02}
    _TEST_K: List[int] = [32, 128, 256]

    @classmethod
    def apply(cls, inp: Inputs, needs_gradient: bool) -> Tuple[torch.Tensor, Optional[Context]]:
        if type(inp.attn_bias) not in FwOp.SUPPORTED_ATTN_BIAS_TYPES:
            raise NotImplementedError('Unsupported attn_bias type')
        if inp.query.ndim in [3, 4]:
            return cls.apply_bmhk(inp, needs_gradient=needs_gradient)
        assert inp.query.ndim == 5, f'query has shape {inp.query.shape}'
        ctx: Optional[Context] = None
        if inp.query.ndim == 5 and inp.query.shape[3] == 1:
            slice_op = partial(torch.squeeze, dim=3)
            inp = replace(inp, query=slice_op(inp.query), key=slice_op(inp.key), value=slice_op(inp.value), attn_bias=_attn_bias_apply(inp.attn_bias, partial(torch.squeeze, dim=2)))
            out, ctx = cls.apply_bmhk(inp, needs_gradient=needs_gradient)
            out = out.unsqueeze(3)
            if ctx is not None:
                ctx = replace(ctx, lse=ctx.lse.unsqueeze(1), out=out)
            return (out, ctx)
        n_groups = inp.key.shape[2]
        main_stream = torch.cuda.current_stream()
        streams = [main_stream] + [torch.cuda.Stream(device=inp.query.device) for _ in range(n_groups - 1)]
        outs = []
        for group, stream in enumerate(streams):
            stream.wait_stream(main_stream)
            with torch.cuda.stream(stream):
                query = inp.query[:, :, group]
                key = inp.key[:, :, group]
                value = inp.value[:, :, group]
                bias = _attn_bias_apply(inp.attn_bias, partial(torch.select, dim=1, index=group))
                outs.append(cls.apply_bmhk(replace(inp, query=query, key=key, value=value, attn_bias=bias), needs_gradient=needs_gradient))
        for s in streams[1:]:
            main_stream.wait_stream(s)
        out = torch.stack([o[0] for o in outs], dim=2)
        if needs_gradient:
            ctx = Context(out=out, lse=torch.stack([o[1].lse for o in outs], dim=1), op_bw=outs[0][1].op_bw)
        return (out, ctx)

    @classmethod
    def apply_bmhk(cls, inp: Inputs, needs_gradient: bool) -> Tuple[torch.Tensor, Optional[Context]]:
        if type(inp.attn_bias) not in FwOp.SUPPORTED_ATTN_BIAS_TYPES:
            raise NotImplementedError('Unsupported attn_bias type')
        seqstart_k, seqstart_q, max_seqlen_q = _get_seqlen_info(inp)
        out, lse, rng_seed, rng_offset = cls.OPERATOR(query=inp.query, key=inp.key, value=inp.value, attn_bias=_get_tensor_bias(inp.attn_bias), seqstart_q=seqstart_q, seqstart_k=seqstart_k, max_seqlen_q=max_seqlen_q, dropout_p=inp.p, compute_logsumexp=needs_gradient, custom_mask_type=_custom_mask_type(inp.attn_bias), scale=inp.scale, seqlen_k=inp.attn_bias.k_seqinfo.seqlen if isinstance(inp.attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask) else None, window_size=inp.attn_bias._window_size if isinstance(inp.attn_bias, (BlockDiagonalCausalLocalAttentionMask, BlockDiagonalCausalLocalAttentionFromBottomRightMask, LowerTriangularFromBottomRightLocalAttentionMask)) else None)
        ctx: Optional[Context] = None
        if needs_gradient:
            ctx = Context(out=out, lse=lse, op_bw=BwOp if inp.p != 0 else None)
            if inp.p != 0:
                ctx.rng_state = torch.tensor([rng_seed, rng_offset], dtype=torch.int64, device='cpu')
        return (out, ctx)

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        matmul_alignment_mn = _minimum_gemm_alignment(d)
        check_lastdim_alignment_stride1(reasons, 'query', d.query, matmul_alignment_mn)
        check_lastdim_alignment_stride1(reasons, 'value', d.value, matmul_alignment_mn)
        _check_bias_alignment(reasons, d.attn_bias)
        _check_large_shapes(reasons, d)
        requires_grad = d.query.requires_grad or d.key.requires_grad or d.value.requires_grad
        if requires_grad:
            reasons.append('Gradience is currently not supported by ck-tiled!')
        return reasons

    @classmethod
    def operator_flop(cls, q, k, v, b, seqstart_q, seqstart_k, max_seqlen_q_, compute_lse, custom_mask_type, *a) -> int:
        return cls.attn_operator_flop(q, k, v, causal=custom_mask_type > 0, seqstart_k=seqstart_k, seqstart_q=seqstart_q)