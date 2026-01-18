import operator
from typing import List
import torch
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
import torch.nn as nn
import torch.nn.functional as F
from ..fuser_method_mappings import (
from ._common_operator_config_utils import _Conv2dMetadata
from .backend_config import (
from .qnnpack import (
def _get_share_qparams_ops_configs() -> List[BackendPatternConfig]:
    """
    Return the operator configs for the operators that works for both float and quantized
    input if input is quantized, the output Tensor shares the same quantization parameter
    with input.

    Example operator: avgpool2d, reshape, transpose, maxpool2d
    Example observed operator:
    observer_0 - avgpool2d - observer_0 (same observer instance as input)
    """
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [qnnpack_default_op_qint8_symmetric_dtype_config, executorch_default_op_quint8_dtype_config]
    share_qparams_ops = [torch.nn.Flatten, F.adaptive_avg_pool2d, F.elu, F.hardtanh, F.max_pool2d, F.pad, F.relu, F.relu6, F.leaky_relu, F.leaky_relu_, torch.nn.AdaptiveAvgPool2d, torch.nn.ConstantPad2d, torch.nn.ELU, torch.nn.MaxPool2d, torch.nn.ReLU6, torch.nn.Hardtanh, torch.nn.LeakyReLU, torch.clamp, torch.flatten, torch.mean, torch.permute, torch.permute_copy, torch.squeeze, 'clamp', 'mean', 'permute', 'reshape', 'relu', 'relu_', 'squeeze', 'squeeze_', 'leaky_relu']
    share_qparams_op_configs: List[BackendPatternConfig] = []
    for op in share_qparams_ops:
        share_qparams_op_configs.append(BackendPatternConfig(op).set_observation_type(observation_type).set_dtype_configs(dtype_configs))
    return share_qparams_op_configs