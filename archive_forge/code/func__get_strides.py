import itertools
from typing import List, Tuple
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
def _get_strides(ts: List[List[torch.Tensor]], tensor_name, dim_0_name, dim_1_name) -> Tuple[List[int], List[int]]:
    strides_0 = [_check_row_or_column('column', idx, tensor_name, dim_0_name, [y.stride(0) for y in x]) for idx, x in enumerate(zip(*ts))]
    strides_1 = [_check_row_or_column('row', idx, tensor_name, dim_1_name, [y.stride(1) for y in x]) for idx, x in enumerate(ts)]
    assert all((s == 1 for s in strides_0)) or all((s == 1 for s in strides_1))
    while len(strides_0) < 3:
        strides_0.append(1 if strides_0[0] == 1 else 0)
    while len(strides_1) < 3:
        strides_1.append(1 if strides_1[0] == 1 else 0)
    return (strides_0, strides_1)