import torch
from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from typing import Tuple
def _quant_min_max_bounds_check(quant_min, quant_max, dtype):
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f'Unsupported dtype: {dtype}')
    quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]
    assert quant_min >= quant_min_lower_bound, f'quant_min out of bound for dtype, quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}'
    assert quant_max <= quant_max_upper_bound, f'quant_max out of bound for dtype, quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}'