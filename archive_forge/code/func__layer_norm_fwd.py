import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
import triton
import triton.language as tl
def _layer_norm_fwd(x, weight, bias, eps, residual=None, x1=None, weight1=None, bias1=None, dropout_p=0.0, rowscale=None, out_dtype=None, residual_dtype=None, is_rms_norm=False, return_dropout_mask=False):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if x1 is not None:
        assert x1.shape == x.shape
        assert rowscale is None
        assert x1.stride(-1) == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.stride(-1) == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.stride(-1) == 1
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    assert y.stride(-1) == 1
    if weight1 is not None:
        y1 = torch.empty_like(y)
        assert y1.stride(-1) == 1
    else:
        y1 = None
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype) or dropout_p > 0.0 or (rowscale is not None) or (x1 is not None):
        residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype if residual_dtype is not None else x.dtype)
        assert residual_out.stride(-1) == 1
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    if dropout_p > 0.0:
        seeds = torch.randint(2 ** 32, (M if x1 is None else 2 * M,), device=x.device, dtype=torch.int64)
    else:
        seeds = None
    if return_dropout_mask and dropout_p > 0.0:
        dropout_mask = torch.empty(M if x1 is None else 2 * M, N, device=x.device, dtype=torch.bool)
    else:
        dropout_mask = None
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[M,](x, y, weight, bias, residual, x1, weight1, bias1, y1, residual_out, rowscale, seeds, dropout_mask, mean, rstd, x.stride(0), y.stride(0), residual.stride(0) if residual is not None else 0, residual_out.stride(0) if residual_out is not None else 0, x1.stride(0) if x1 is not None else 0, y1.stride(0) if y1 is not None else 0, M, N, eps, dropout_p, is_rms_norm, BLOCK_N, residual is not None, residual_out is not None, bias is not None, dropout_p > 0.0, dropout_mask is not None, rowscale is not None)
    if dropout_mask is not None and x1 is not None:
        dropout_mask, dropout_mask1 = dropout_mask.tensor_split(2, dim=0)
    else:
        dropout_mask1 = None
    return (y, y1, mean, rstd, residual_out if residual_out is not None else x, seeds, dropout_mask, dropout_mask1)