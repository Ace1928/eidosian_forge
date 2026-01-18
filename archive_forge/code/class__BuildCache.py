import numbers
from functools import reduce
from operator import mul
import numpy as np
class _BuildCache:

    def __init__(self, arr_seq, common_shape, dtype):
        self.offsets = list(arr_seq._offsets)
        self.lengths = list(arr_seq._lengths)
        self.next_offset = arr_seq._get_next_offset()
        self.bytes_per_buf = arr_seq._buffer_size * MEGABYTE
        self.dtype = dtype if arr_seq._data.size == 0 else arr_seq._data.dtype
        if arr_seq.common_shape != () and common_shape != arr_seq.common_shape:
            raise ValueError('All dimensions, except the first one, must match exactly')
        self.common_shape = common_shape
        n_in_row = reduce(mul, common_shape, 1)
        bytes_per_row = n_in_row * dtype.itemsize
        self.rows_per_buf = max(1, self.bytes_per_buf // bytes_per_row)

    def update_seq(self, arr_seq):
        arr_seq._offsets = np.array(self.offsets)
        arr_seq._lengths = np.array(self.lengths)