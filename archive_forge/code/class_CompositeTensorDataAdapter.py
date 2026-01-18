import abc
import contextlib
import functools
import itertools
import math
import random
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import dataset_creator
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
class CompositeTensorDataAdapter(DataAdapter):
    """Adapter that handles composite tensor."""

    @staticmethod
    def can_handle(x, y=None):
        flat_inputs = nest.flatten(x)
        if y is not None:
            flat_inputs += nest.flatten(y)

        def _is_composite(v):
            if tf_utils.is_extension_type(v) and (not isinstance(v, (dataset_ops.DatasetV2, iterator_ops.IteratorBase))) and (not _is_distributed_dataset(v)):
                return True
            return _is_scipy_sparse(v)

        def _is_tensor_or_composite(v):
            if isinstance(v, (tensor.Tensor, np.ndarray)):
                return True
            return _is_composite(v)
        return any((_is_composite(v) for v in flat_inputs)) and all((_is_tensor_or_composite(v) for v in flat_inputs))

    def __init__(self, x, y=None, sample_weights=None, sample_weight_modes=None, batch_size=None, steps=None, shuffle=False, **kwargs):
        super(CompositeTensorDataAdapter, self).__init__(x, y, **kwargs)
        x, y, sample_weights = _process_tensorlike((x, y, sample_weights))
        sample_weight_modes = broadcast_sample_weight_modes(sample_weights, sample_weight_modes)
        sample_weights, _, _ = training_utils.handle_partial_sample_weights(y, sample_weights, sample_weight_modes, check_all_flat=True)
        inputs = pack_x_y_sample_weight(x, y, sample_weights)
        dataset = dataset_ops.DatasetV2.from_tensor_slices(inputs)
        num_samples = int(nest.flatten(x)[0].shape[0])
        if shuffle:
            dataset = dataset.shuffle(num_samples)
        if not batch_size:
            batch_size = int(math.ceil(num_samples / steps)) if steps else 32
        dataset = dataset.batch(batch_size)
        self._size = int(math.ceil(num_samples / batch_size))
        self._batch_size = batch_size
        self._has_partial_batch = self._size != num_samples // batch_size
        self._partial_batch_size = None
        if self._has_partial_batch:
            self._partial_batch_size = num_samples - (self._size - 1) * self._batch_size
        self._dataset = dataset

    def get_dataset(self):
        return self._dataset

    def get_size(self):
        return self._size

    def batch_size(self):
        return self._batch_size

    def has_partial_batch(self):
        return self._has_partial_batch

    def partial_batch_size(self):
        return self._partial_batch_size

    def should_recreate_iterator(self):
        return True