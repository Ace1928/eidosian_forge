import copy
import operator
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
from collections import namedtuple
from typing import Callable, Dict, List, Union
from .backend_config import (
from ..fuser_method_mappings import (
def _add_fixed_qparams_to_dtype_configs(dtype_configs: List[DTypeConfig], constraints: DTypeWithConstraints) -> List[DTypeConfig]:
    """
    Return a copy of the list of DTypeConfigs where activations are subject to the specified
    constraints required for fixed qparams ops.

    If the data type doesn't match the one in the constraints, simply leave the corresponding
    DTypeConfig unchanged.

    If `scale_min_lower_bound` or `scale_max_upper_bound` is specified in the activations,
    throw an exception since these settings are incompatible with fixed qparams ops.
    """
    new_dtype_configs = []
    for dtype_config in dtype_configs:
        dc = copy.deepcopy(dtype_config)
        for orig_constraints in [dc.input_dtype_with_constraints, dc.output_dtype_with_constraints]:
            if orig_constraints.dtype != constraints.dtype:
                continue
            if orig_constraints.scale_min_lower_bound is not None:
                raise ValueError(f'scale_min_lower_bound is invalid for fixed qparams ops: {dtype_config}')
            if orig_constraints.scale_max_upper_bound is not None:
                raise ValueError(f'scale_max_upper_bound is invalid for fixed qparams ops: {dtype_config}')
            orig_constraints.quant_min_lower_bound = constraints.quant_min_lower_bound
            orig_constraints.quant_max_upper_bound = constraints.quant_max_upper_bound
            orig_constraints.scale_exact_match = constraints.scale_exact_match
            orig_constraints.zero_point_exact_match = constraints.zero_point_exact_match
        new_dtype_configs.append(dc)
    return new_dtype_configs