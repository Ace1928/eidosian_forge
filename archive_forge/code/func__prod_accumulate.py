import triton
import triton.language as tl
@triton.jit
def _prod_accumulate(a, b):
    return a * b