import logging
from typing import Optional
import torch
import torch.nn as nn
import triton
from torch.cuda.amp import custom_bwd, custom_fwd
from xformers.triton.k_layer_norm import (
class _LayerNorm(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16 if _triton_layernorm_fp16_enabled else None)
    def forward(ctx, x, weight, bias, eps):
        if x.dtype == torch.float16:
            eps = max(eps, 1.6e-05)
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device='cuda')
        rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        if not x_arg.is_contiguous() or not y.is_contiguous():
            global _triton_registered_warnings
            if not _triton_registered_warnings:
                logger.warning('Non-contiguous input tensor found. Making it contiguous,' + ' but could have perf or trainer implications')
                _triton_registered_warnings = True
            x_arg = x_arg.contiguous()
            y = y.contiguous()
        num_warps = min(max(BLOCK_SIZE_N // 256, 1), 16)
        layer_norm_fw[M,](x_arg, y, weight, bias, mean, rstd, x_arg.stride(0), N, eps, num_warps=num_warps, BLOCK_SIZE_N=BLOCK_SIZE_N, affine=weight is not None)
        ctx.save_for_backward(x, mean, rstd, weight)
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.num_warps = num_warps
        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        x, mean, rstd, weight = ctx.saved_tensors
        x = x.reshape(-1, x.size(-1))
        M, N = x.size()
        GROUP_SIZE_M = 32
        if N <= 8192:
            GROUP_SIZE_M = 64
        if N <= 4096:
            GROUP_SIZE_M = 96
        if N <= 2048:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256
        if dy.dtype == torch.float32:
            GROUP_SIZE_M = GROUP_SIZE_M // 2
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device='cuda')
        t_args = {'dtype': x.dtype, 'device': x.device}
        _dw = torch.empty((GROUP_SIZE_M, x.size(-1)), **t_args)
        _db = torch.empty_like(_dw)
        dw = torch.empty((x.size(-1),), **t_args)
        db = torch.empty_like(dw)
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        assert dy.numel() == x.numel(), 'Something is wrong in the backward graph, possibly because of an inplace operation after the layernorm'
        num_warps = min(max(ctx.BLOCK_SIZE_N // 256, 1), 16)
        layer_norm_bwd_dx_fused[M,](dx, dy, _dw, _db, x, weight if weight is not None else x, mean, rstd, locks, x.stride(0), N, affine=weight is not None, GROUP_SIZE_M=GROUP_SIZE_M, BLOCK_SIZE_N=ctx.BLOCK_SIZE_N, num_warps=num_warps)

        def grid(meta):
            return [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        layer_norm_bwd_dwdb[grid](_dw, _db, dw, db, GROUP_SIZE_M, N, BLOCK_SIZE_M=32, BLOCK_SIZE_N=64)
        dx = dx.reshape_as(dy)
        return (dx, dw, db, None)