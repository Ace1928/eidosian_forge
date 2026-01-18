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
def _build_model_with_inputs(self, inputs, targets):
    """Build the model (set model inputs/outputs), mainly for subclass model."""
    processed_inputs = []
    is_dict_inputs = False
    orig_inputs = inputs
    if isinstance(inputs, (data_types.DatasetV1, data_types.DatasetV2)):
        inputs, targets, _ = training_utils_v1.extract_tensors_from_dataset(inputs)
    training_utils_v1.validate_input_types(inputs, orig_inputs)
    if isinstance(inputs, (list, tuple)):
        processed_inputs += list(inputs)
    elif isinstance(inputs, dict):
        is_dict_inputs = True
        keys = sorted(inputs.keys())
        processed_inputs = [inputs[k] for k in keys]
    else:
        processed_inputs.append(inputs)
    for input_tensor in processed_inputs:
        if training_utils_v1.is_composite_or_composite_value(input_tensor):
            raise ValueError('All SparseTensor and RaggedTensor inputs must be explicitly declared using a keras.Input() with sparse=True or ragged=True. We found an undeclared input %s. For Sequential models, please add a keras.Input() as your first Layer. For subclassed models, please call self._set_inputs() on your input set, which you can create using keras.Input() for each input to your model.' % (input_tensor,))
    if isinstance(orig_inputs, (data_types.DatasetV1, data_types.DatasetV2, iterator_ops.Iterator)):
        if not self.inputs:
            inputs = training_utils_v1.cast_if_floating_dtype(inputs, self.dtype)

        def create_tensor_spec(t):
            return tensor_spec.TensorSpec(t.shape, t.dtype)
        cast_inputs = nest.map_structure(create_tensor_spec, inputs)
    elif training_utils_v1.has_tensors(inputs):
        cast_inputs = training_utils_v1.cast_if_floating_dtype(inputs)
    else:
        cast_inputs = inputs
    self._set_inputs(cast_inputs)
    return (processed_inputs, targets, is_dict_inputs)