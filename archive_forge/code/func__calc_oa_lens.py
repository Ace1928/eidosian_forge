import math
import cupy
from cupy._core import internal
from cupyx.scipy import fft
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.ndimage import _util
def _calc_oa_lens(s1, s2):
    fallback = (s1 + s2 - 1, None, s1, s2)
    if s1 == s2 or s1 == 1 or s2 == 1:
        return fallback
    swapped = s2 > s1
    if swapped:
        s1, s2 = (s2, s1)
    if s2 >= s1 // 2:
        return fallback
    overlap = s2 - 1
    block_size = fft.next_fast_len(_optimal_oa_block_size(overlap))
    if block_size >= s1:
        return fallback
    in1_step, in2_step = (block_size - s2 + 1, s2)
    if swapped:
        in1_step, in2_step = (in2_step, in1_step)
    return (block_size, overlap, in1_step, in2_step)