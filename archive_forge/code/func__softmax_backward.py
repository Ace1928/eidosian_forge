import triton
import triton.language as tl
@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16)], key=['K'])
@triton.jit
def _softmax_backward(GradIn, GradOut, Out, stride_bm, stride_bn, stride_gm, stride_gn, stride_om, stride_on, K, depth: tl.constexpr, causal: tl.constexpr, log: tl.constexpr):
    """
    Compute the softmax gradients.
    ..Note: Not autotuning for now because this would lead to broken accumulated gradients
    """
    m = tl.program_id(0)
    n = tl.program_id(1)
    k = tl.arange(0, depth)
    grad_out_ptrs = GradOut + m * stride_gm + n * stride_gn + k
    out_ptrs = Out + m * stride_om + n * stride_on + k
    io_mask = k < K
    if causal:
        io_mask = io_mask & (k <= n)
    g = tl.load(grad_out_ptrs, mask=io_mask, other=float(0)).to(tl.float32)
    o = tl.load(out_ptrs, mask=io_mask, other=float(0)).to(tl.float32)
    if causal:
        zero = float(0)
        zero = zero.to(g.dtype)
        g = tl.where(k > n, zero, g)
        o = tl.where(k > n, zero, o)
    if log:
        s = tl.sum(g, 0)
        grad_in = g - tl.exp(o) * s
    else:
        s = tl.sum(g * o, 0)
        grad_in = o * (g - s)
    grad_in_ptrs = GradIn + m * stride_bm + n * stride_bn + k
    tl.store(grad_in_ptrs, grad_in, mask=k < K)