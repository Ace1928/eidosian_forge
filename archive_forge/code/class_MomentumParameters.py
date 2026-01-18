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
class MomentumParameters(_OptimizationParameters):
    """Optimization parameters for Momentum with TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=tf.tpu.experimental.MomentumParameters(0.1),
          ...))
  ```

  """

    def __init__(self, learning_rate: float, momentum: float, use_nesterov: bool=False, use_gradient_accumulation: bool=True, clip_weight_min: Optional[float]=None, clip_weight_max: Optional[float]=None, weight_decay_factor: Optional[float]=None, multiply_weight_decay_factor_by_learning_rate: Optional[bool]=None, clip_gradient_min: Optional[float]=None, clip_gradient_max: Optional[float]=None):
        """Optimization parameters for momentum.

    Args:
      learning_rate: a floating point value. The learning rate.
      momentum: a floating point value.  The momentum.
      use_nesterov: If `True` use Nesterov Momentum. See (Sutskever et al.,
        2013). This implementation always computes gradients at the value of the
        variable(s) passed to the optimizer. Using Nesterov Momentum makes the
        variable(s) track the values called `theta_t + mu*v_t` in the paper.
        This implementation is an approximation of the original formula, valid
        for high values of momentum. It will compute the "adjusted gradient" in
        NAG by assuming that the new gradient will be estimated by the current
        average gradient plus the product of momentum and the change in the
        average gradient.
      use_gradient_accumulation: setting this to `False` makes embedding
        gradients calculation less accurate but faster. Please see
        `optimization_parameters.proto` for details.
      clip_weight_min: the minimum value to clip by; None means -infinity.
      clip_weight_max: the maximum value to clip by; None means +infinity.
      weight_decay_factor: amount of weight decay to apply; None means that the
        weights are not decayed.
      multiply_weight_decay_factor_by_learning_rate: if true,
        `weight_decay_factor` is multiplied by the current learning rate.
      clip_gradient_min: the minimum value to clip by; None means -infinity.
        Gradient accumulation must be set to true if this is set.
      clip_gradient_max: the maximum value to clip by; None means +infinity.
        Gradient accumulation must be set to true if this is set.
    """
        super().__init__(learning_rate=learning_rate, use_gradient_accumulation=use_gradient_accumulation, clip_weight_min=clip_weight_min, clip_weight_max=clip_weight_max, weight_decay_factor=weight_decay_factor, multiply_weight_decay_factor_by_learning_rate=multiply_weight_decay_factor_by_learning_rate, clip_gradient_min=clip_gradient_min, clip_gradient_max=clip_gradient_max)
        self.momentum = momentum
        self.use_nesterov = use_nesterov