from math import ceil
import cupy
def _output_len(len_h, in_len, up, down):
    return ((in_len - 1) * up + len_h - 1) // down + 1