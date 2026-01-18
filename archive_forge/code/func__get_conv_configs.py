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
def _get_conv_configs() -> List[BackendPatternConfig]:
    """
    Return all configs related to conv modules and ops.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [qnnpack_weighted_op_qint8_symmetric_dtype_config, executorch_weighted_op_int8_dtype_config]
    conv_configs = []
    for convs in [_Conv2dMetadata]:
        conv_configs.append(BackendPatternConfig(convs.root).set_observation_type(observation_type).set_dtype_configs(dtype_configs).set_root_module(convs.root).set_reference_quantized_module(convs.reference).set_qat_module(convs.qat))
        conv_configs.append(BackendPatternConfig(convs.qat).set_observation_type(observation_type).set_dtype_configs(dtype_configs).set_root_module(convs.root).set_reference_quantized_module(convs.reference))
        conv_configs.append(BackendPatternConfig(convs.func).set_observation_type(observation_type).set_dtype_configs(dtype_configs)._set_input_type_to_index({'weight': 1, 'bias': 2}))
        conv_configs.append(BackendPatternConfig((convs.root, nn.ReLU)).set_dtype_configs(dtype_configs).set_fuser_method(_sequential_wrapper2(convs.fused_conv_relu)).set_fused_module(convs.fused_conv_relu))
        conv_configs.append(BackendPatternConfig((convs.root, F.relu)).set_dtype_configs(dtype_configs).set_fuser_method(_sequential_wrapper2(convs.fused_conv_relu)).set_fused_module(convs.fused_conv_relu))
        conv_configs.append(BackendPatternConfig(convs.fused_conv_relu).set_observation_type(observation_type).set_dtype_configs(dtype_configs).set_root_module(convs.root).set_reference_quantized_module(convs.reference).set_qat_module(convs.relu_qat))
        conv_configs.append(BackendPatternConfig(convs.relu_qat).set_observation_type(observation_type).set_dtype_configs(dtype_configs).set_root_module(convs.root).set_reference_quantized_module(convs.reference))
        conv_configs.append(BackendPatternConfig((convs.func, nn.ReLU)).set_observation_type(observation_type).set_dtype_configs(dtype_configs))
        conv_configs.append(BackendPatternConfig((convs.func, F.relu)).set_observation_type(observation_type).set_dtype_configs(dtype_configs))
        conv_configs.append(BackendPatternConfig(convs.fused_conv_relu).set_dtype_configs(dtype_configs).set_qat_module(convs.relu_qat))
        conv_configs.append(BackendPatternConfig(convs.relu_qat).set_dtype_configs(dtype_configs).set_root_module(convs.root).set_reference_quantized_module(convs.reference))
        conv_configs.append(BackendPatternConfig((convs.root, convs.bn)).set_dtype_configs(dtype_configs).set_fuser_method(fuse_conv_bn).set_fused_module(convs.fused_conv_bn))
        conv_configs.append(BackendPatternConfig((convs.root, convs.bn, nn.ReLU)).set_dtype_configs(dtype_configs).set_fuser_method(fuse_conv_bn_relu).set_fused_module(convs.fused_conv_bn_relu))
        conv_configs.append(BackendPatternConfig((convs.root, convs.bn, F.relu)).set_dtype_configs(dtype_configs).set_root_module(convs.root).set_fuser_method(fuse_conv_bn_relu).set_fused_module(convs.fused_conv_bn_relu))
        conv_configs.append(BackendPatternConfig(convs.fused_conv_bn).set_dtype_configs(dtype_configs).set_qat_module(convs.bn_qat))
        conv_configs.append(BackendPatternConfig(convs.fused_conv_bn_relu).set_dtype_configs(dtype_configs).set_qat_module(convs.bn_relu_qat))
        conv_configs.append(BackendPatternConfig(convs.bn_qat).set_observation_type(observation_type).set_dtype_configs(dtype_configs).set_root_module(convs.root).set_reference_quantized_module(convs.reference))
        conv_configs.append(BackendPatternConfig(convs.bn_relu_qat).set_observation_type(observation_type).set_dtype_configs(dtype_configs).set_root_module(convs.root).set_reference_quantized_module(convs.reference))
    return conv_configs