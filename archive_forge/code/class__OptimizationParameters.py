import collections
import copy
import math
import re
from typing import Optional
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util.tf_export import tf_export
class _OptimizationParameters:
    """Parameters common to all optimizations."""

    def __init__(self, learning_rate: float, use_gradient_accumulation: bool, clip_weight_min: Optional[float], clip_weight_max: Optional[float], weight_decay_factor: Optional[float], multiply_weight_decay_factor_by_learning_rate: Optional[bool], clip_gradient_min: Optional[float]=None, clip_gradient_max: Optional[float]=None):
        self.learning_rate = learning_rate
        self.use_gradient_accumulation = use_gradient_accumulation
        self.clip_weight_min = clip_weight_min
        self.clip_weight_max = clip_weight_max
        self.weight_decay_factor = weight_decay_factor
        self.multiply_weight_decay_factor_by_learning_rate = multiply_weight_decay_factor_by_learning_rate
        self.clip_gradient_min = clip_gradient_min
        self.clip_gradient_max = clip_gradient_max
        if not use_gradient_accumulation and (clip_gradient_min is not None or clip_gradient_max is not None):
            raise ValueError('When using gradient clipping limits, gradient  accumulation must be enabled.')