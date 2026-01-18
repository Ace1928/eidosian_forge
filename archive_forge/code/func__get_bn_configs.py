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
def _get_bn_configs() -> List[BackendPatternConfig]:
    """
    Return all configs related to batchnorm.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [qnnpack_default_op_qint8_symmetric_dtype_config, executorch_default_op_quint8_dtype_config]
    bn_configs = []
    bn_configs.append(BackendPatternConfig(nn.BatchNorm2d).set_observation_type(observation_type).set_dtype_configs(dtype_configs))
    return bn_configs