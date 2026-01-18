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
def _validate_or_infer_batch_size(self, batch_size, steps, x):
    """Validates that the `batch_size` provided is consistent with InputLayer.

    It's possible that the user specified a static batch size in their
    InputLayer. If so, this method checks the provided `batch_size` and `x`
    arguments are consistent with this static batch size. Also, if
    `batch_size` is `None`, this method will attempt to infer the batch size
    from the static batch size of the InputLayer. Lastly, ValueError will be
    raised if `x` is a tf.data.Dataset and `batch_size` is specified as we
    expect users to provide batched datasets.

    Args:
      batch_size: The batch_size provided as an argument to
        fit/evaluate/predict.
      steps: The steps provided as an argument to fit/evaluate/predict.
      x: The data passed as `x` to fit/evaluate/predict.

    Returns:
      The validated batch_size, auto-inferred from the first layer if not
      provided.
    """
    if isinstance(x, (data_types.DatasetV1, data_types.DatasetV2, data_utils.Sequence)) or tf_inspect.isgenerator(x):
        if batch_size is not None:
            raise ValueError('The `batch_size` argument must not be specified for the given input type. Received input: {}, batch_size: {}'.format(x, batch_size))
        return
    layers = self._flatten_layers(include_self=False, recursive=False)
    first_layer = next(layers, None)
    if first_layer:
        static_batch_size = training_utils.get_static_batch_size(first_layer)
        if static_batch_size is not None:
            if self._distribution_strategy and distributed_training_utils.global_batch_size_supported(self._distribution_strategy):
                num_splits_for_ds = self._distribution_strategy.num_replicas_in_sync
            else:
                num_splits_for_ds = 1
            if batch_size is not None:
                if batch_size % num_splits_for_ds != 0:
                    raise ValueError('The `batch_size` argument ({}) must be divisible the by number of replicas ({})'.format(batch_size, num_splits_for_ds))
                per_replica_batch_size = batch_size // num_splits_for_ds
                if per_replica_batch_size != static_batch_size:
                    raise ValueError('The `batch_size` argument value {} is incompatible with the specified batch size of your Input Layer: {}'.format(per_replica_batch_size, static_batch_size))
            if isinstance(x, (data_types.DatasetV2, iterator_ops.Iterator, iterator_ops.IteratorBase)):
                ds_batch_size = tensor_shape.Dimension(nest.flatten(dataset_ops.get_legacy_output_shapes(x))[0][0]).value
                if ds_batch_size is not None:
                    if ds_batch_size % num_splits_for_ds != 0:
                        raise ValueError('The batch output shape of your `Dataset` {} cannot be divisible by number of replicas {}'.format(ds_batch_size, num_splits_for_ds))
                    ds_per_replica_batch_size = ds_batch_size // num_splits_for_ds
                    if ds_per_replica_batch_size != static_batch_size:
                        raise ValueError('The batch output shape of your `Dataset` is {}, which is incompatible with the specified batch size of your Input Layer: {}'.format(ds_per_replica_batch_size, static_batch_size))
            if steps is None:
                batch_size = static_batch_size * num_splits_for_ds
    if batch_size is None and steps is None:
        batch_size = 32
    return batch_size