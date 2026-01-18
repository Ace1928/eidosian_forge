import itertools
from typing import List, Tuple
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
def _check_row_or_column(row_or_col_type, row_or_col_idx, tensor_name, dim_name, vals):
    assert len(vals) > 0
    for pos, val in enumerate(vals[1:]):
        assert val == vals[0], f'the tensors on {row_or_col_type} {row_or_col_idx} of the {tensor_name} must all have the same stride along the {dim_name} dimension, got {vals[0]} at position 0 and {val} at position {pos + 1}'
    return vals[0]