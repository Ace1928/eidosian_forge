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
def _standardize_tensors(self, x, y, sample_weight, run_eagerly, dict_inputs, is_dataset, class_weight=None, batch_size=None):
    if run_eagerly:
        feed_input_names = self.input_names
        feed_input_shapes = None
    elif not self._is_graph_network:
        feed_input_names = self._feed_input_names
        feed_input_shapes = None
    else:
        feed_input_names = self._feed_input_names
        feed_input_shapes = self._feed_input_shapes
    if not isinstance(x, (data_types.DatasetV1, data_types.DatasetV2)):
        x = training_utils_v1.standardize_input_data(x, feed_input_names, feed_input_shapes, check_batch_axis=False, exception_prefix='input')
    if isinstance(x, data_types.DatasetV2):
        x_shapes = dataset_ops.get_structure(x)
        if isinstance(x_shapes, tuple):
            x_shapes = x_shapes[0]
    else:
        flat_inputs = nest.flatten(x, expand_composites=False)
        flat_expected_inputs = nest.flatten(self.inputs, expand_composites=False)
        converted_x = []
        for a, b in zip(flat_inputs, flat_expected_inputs):
            converted_x.append(_convert_scipy_sparse_tensor(a, b))
        x = nest.pack_sequence_as(x, converted_x, expand_composites=False)

        def _type_spec_from_value(value):
            """Grab type_spec without converting array-likes to tensors."""
            if tf_utils.is_extension_type(value):
                return value._type_spec
            if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                return tensor_spec.TensorSpec(value.shape, value.dtype)
            else:
                return type_spec.type_spec_from_value(value)
        x_shapes = nest.map_structure(_type_spec_from_value, x)
    flat_inputs = nest.flatten(x_shapes, expand_composites=False)
    flat_expected_inputs = nest.flatten(self.inputs, expand_composites=False)
    for a, b in zip(flat_inputs, flat_expected_inputs):
        nest.assert_same_structure(a, b, expand_composites=True)
    if y is not None:
        training_utils_v1.prepare_sample_weight_modes(self._training_endpoints, self.sample_weight_mode)
        feed_output_names = self._feed_output_names
        feed_sample_weight_modes = self._sample_weight_modes
        if not self._is_graph_network:
            feed_output_shapes = None
        else:
            feed_output_shapes = self._feed_output_shapes
        y = training_utils_v1.standardize_input_data(y, feed_output_names, shapes=None, check_batch_axis=False, exception_prefix='target')
        sample_weights = training_utils_v1.standardize_sample_weights(sample_weight, feed_output_names)
        class_weights = training_utils_v1.standardize_class_weights(class_weight, feed_output_names)
        sample_weights = [training_utils_v1.standardize_weights(ref, sw, cw, mode) for ref, sw, cw, mode in zip(y, sample_weights, class_weights, feed_sample_weight_modes)]
        if not self._distribution_strategy:
            training_utils_v1.check_array_lengths(x, y, sample_weights)
            if self._is_graph_network and (not run_eagerly):
                training_utils_v1.check_loss_and_target_compatibility(y, self._feed_loss_fns, feed_output_shapes)
        sample_weights, _, _ = training_utils.handle_partial_sample_weights(y, sample_weights, feed_sample_weight_modes, check_all_flat=True)
    else:
        y = []
        sample_weights = None
    if self.stateful and batch_size and (not is_dataset):
        if x[0].shape[0] % batch_size != 0:
            raise ValueError('In a stateful network, you should only pass inputs with a number of samples that can be divided by the batch size. Found: ' + str(x[0].shape[0]) + ' samples')
    if dict_inputs and (not isinstance(x, (data_types.DatasetV1, data_types.DatasetV2))):
        x = dict(zip(feed_input_names, x))
    return (x, y, sample_weights)