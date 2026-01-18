import collections
import warnings
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_distributed_v1
from tensorflow.python.keras.engine import training_eager_v1
from tensorflow.python.keras.engine import training_generator_v1
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def _set_optimizer(self, optimizer):
    """Sets self.optimizer.

    Sets self.optimizer to `optimizer`, potentially wrapping it with a
    LossScaleOptimizer.

    Args:
      optimizer: The optimizer(s) to assign to self.optimizer.
    """
    if isinstance(optimizer, (list, tuple)):
        self.optimizer = [optimizers.get(opt) for opt in optimizer]
    else:
        self.optimizer = optimizers.get(optimizer)
    if isinstance(self._dtype_policy, policy.PolicyV1):
        loss_scale = self._dtype_policy.loss_scale
    elif self._dtype_policy.name == 'mixed_float16':
        loss_scale = 'dynamic'
    else:
        loss_scale = None
    if loss_scale is not None and (not isinstance(self.optimizer, loss_scale_optimizer.LossScaleOptimizer)):
        if isinstance(self.optimizer, list):
            raise ValueError('When a dtype policy with a loss scale is used, you can only pass a single optimizer. Using policy %s and got optimizers: %s' % self._dtype_policy, self.optimizer)
        if not isinstance(self.optimizer, optimizer_v2.OptimizerV2):
            raise ValueError('"optimizer" must be an instance of tf.keras.optimizers.Optimizer when a dype policy with a loss scale  used, but got: %s. Using policy: %s' % (self.optimizer, self._dtype_policy))
        if loss_scale == 'dynamic':
            self.optimizer = loss_scale_optimizer.LossScaleOptimizer(self.optimizer)
        else:
            self.optimizer = loss_scale_optimizer.LossScaleOptimizerV1(self.optimizer, loss_scale)