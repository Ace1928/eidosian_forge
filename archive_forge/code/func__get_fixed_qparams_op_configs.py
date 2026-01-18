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
def _get_fixed_qparams_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    fixed_qparams_op_configs = []
    for fixed_qparam_op, constraints in _FIXED_QPARAMS_OP_TO_CONSTRAINTS.items():
        new_dtype_configs = _add_fixed_qparams_to_dtype_configs(dtype_configs, constraints)
        fixed_qparams_op_configs.append(BackendPatternConfig(fixed_qparam_op).set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT).set_dtype_configs(new_dtype_configs))
    return fixed_qparams_op_configs