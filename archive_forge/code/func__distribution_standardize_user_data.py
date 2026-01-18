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
def _distribution_standardize_user_data(self, x, y=None, sample_weight=None, class_weight=None, batch_size=None, validation_split=0, shuffle=False, epochs=1, allow_partial_batch=False):
    """Runs validation checks on input and target data passed by the user.

    This is called when using tf.distribute.Strategy to train, evaluate or serve
    the model.

    Args:
      x: Input data. A numpy array or `tf.data` dataset.
      y: Target data. A numpy array or None if x is a `tf.data` dataset.
      sample_weight: An optional sample-weight array passed by the user to
        weight the importance of each sample in `x`.
      class_weight: An optional class-weight array by the user to
        weight the importance of samples in `x` based on the class they belong
        to, as conveyed by `y`.
      batch_size: Integer batch size. If provided, it is used to run additional
        validation checks on stateful models.
      validation_split: Float between 0 and 1.
        Fraction of the training data to be used as validation data.
      shuffle: Boolean whether to shuffle the training data before each epoch.
      epochs: Integer epochs. If > 1, repeat the numpy training data epochs
        times when converting to training dataset.
      allow_partial_batch: Boolean whether to enforce that all batches have the
        same size.

    Returns:
      Dataset instance.

    Raises:
      ValueError: In case of invalid user-provided data.
      RuntimeError: If the model was never compiled.
    """
    if class_weight:
        raise NotImplementedError('`class_weight` is currently not supported when using tf.distribute.Strategy.')
    if sample_weight is not None and sample_weight.all() and backend.is_tpu_strategy(self._distribution_strategy):
        raise NotImplementedError('`sample_weight` is currently not supported when using TPUStrategy.')
    if isinstance(x, data_types.DatasetV2):
        if shuffle:
            training_utils_v1.verify_dataset_shuffled(x)
    strategy = self._distribution_strategy
    with strategy.scope():
        if ops.executing_eagerly_outside_functions():
            session = None
        else:
            session = backend.get_session()
        first_x_value = nest.flatten(x)[0]
        if isinstance(first_x_value, np.ndarray):
            x = training_utils.list_to_tuple(x)
            if y is not None:
                y = training_utils.list_to_tuple(y)
                if sample_weight is not None:
                    sample_weight = training_utils.list_to_tuple(sample_weight)
                    in_tuple = (x, y, sample_weight)
                else:
                    in_tuple = (x, y)
            else:
                in_tuple = x
            ds = strategy.extended.experimental_make_numpy_dataset(in_tuple, session=session)
            if shuffle:
                ds = ds.shuffle(max(1024, batch_size * 8))
            if epochs > 1:
                ds = ds.repeat(epochs)
            drop_remainder = not allow_partial_batch and strategy.extended.experimental_require_static_shapes
            if backend.is_tpu_strategy(strategy) and (not drop_remainder):
                dataset_size = first_x_value.shape[0]
                if dataset_size % batch_size == 0:
                    drop_remainder = True
            x = ds.batch(batch_size, drop_remainder=drop_remainder)
        else:
            assert isinstance(x, data_types.DatasetV2)
            training_utils_v1.validate_dataset_input(x, y, sample_weight, validation_split)
    return x