import torch
from bitsandbytes.triton.triton_utils import is_triton_available
@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 1024}, num_warps=4), triton.Config({'BLOCK_SIZE': 2048}, num_stages=1)], key=['n_elements'])
@triton.jit
def _quantize_global(x_ptr, absmax_inv_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)
    output = tl.libdevice.llrint(127.0 * (x * absmax_inv))
    tl.store(output_ptr + offsets, output, mask=mask)