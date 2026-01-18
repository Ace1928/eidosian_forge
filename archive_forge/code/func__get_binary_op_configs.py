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
def _get_binary_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    binary_op_configs: List[BackendPatternConfig] = []
    num_tensor_args_to_observation_type_mapping = {0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT, 1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT, 2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT}
    for op_with_quantized_bop_scalar_variant in [operator.add, torch.add, operator.mul, torch.mul]:
        bop_patterns = [(op_with_quantized_bop_scalar_variant, nn.ReLU), (op_with_quantized_bop_scalar_variant, F.relu), (op_with_quantized_bop_scalar_variant, torch.relu), op_with_quantized_bop_scalar_variant]
        for bop_pattern in bop_patterns:
            binary_op_configs.append(BackendPatternConfig(bop_pattern).set_dtype_configs(dtype_configs)._set_num_tensor_args_to_observation_type(num_tensor_args_to_observation_type_mapping))
    binary_op_configs.append(BackendPatternConfig(torch.matmul).set_dtype_configs(dtype_configs))
    return binary_op_configs