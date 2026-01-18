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
def _compile_from_inputs(self, all_inputs, target, orig_inputs, orig_target):
    if target is not None:
        if training_utils_v1.has_tensors(target):
            target = training_utils_v1.cast_if_floating_dtype_and_mismatch(target, self.outputs)
        training_utils_v1.validate_input_types(target, orig_target, allow_dict=False, field_name='target')
        if isinstance(target, (list, tuple)):
            all_inputs += list(target)
        else:
            all_inputs.append(target)
    if any((tensor_util.is_tf_type(v) for v in all_inputs)):
        if not all((tensor_util.is_tf_type(v) for v in all_inputs)):
            raise ValueError('Do not pass inputs that mix Numpy arrays and TensorFlow tensors. You passed: x=' + str(orig_inputs) + '; y=' + str(orig_target))
    is_dataset = isinstance(orig_inputs, (data_types.DatasetV1, data_types.DatasetV2, iterator_ops.Iterator))
    if is_dataset or context.executing_eagerly():
        target_tensors = None
    elif target is not None:
        if not isinstance(target, (list, tuple)):
            target = [target]
        target_tensors = [v for v in target if _is_symbolic_tensor(v)]
    else:
        target_tensors = None
    self.compile(optimizer=self.optimizer, loss=self.loss, metrics=self._compile_metrics, weighted_metrics=self._compile_weighted_metrics, loss_weights=self.loss_weights, target_tensors=target_tensors, sample_weight_mode=self.sample_weight_mode, run_eagerly=self.run_eagerly, experimental_run_tf_function=self._experimental_run_tf_function)