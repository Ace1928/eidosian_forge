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
class TensorLikeDataAdapter(DataAdapter):
    """Adapter that handles Tensor-like objects, e.g. EagerTensor and NumPy."""

    @staticmethod
    def can_handle(x, y=None):
        flat_inputs = nest.flatten(x)
        if y is not None:
            flat_inputs += nest.flatten(y)
        tensor_types = _get_tensor_types()

        def _is_tensor(v):
            if isinstance(v, tensor_types):
                return True
            return False
        return all((_is_tensor(v) for v in flat_inputs))

    def __init__(self, x, y=None, sample_weights=None, sample_weight_modes=None, batch_size=None, epochs=1, steps=None, shuffle=False, **kwargs):
        super(TensorLikeDataAdapter, self).__init__(x, y, **kwargs)
        x, y, sample_weights = _process_tensorlike((x, y, sample_weights))
        sample_weight_modes = broadcast_sample_weight_modes(sample_weights, sample_weight_modes)
        sample_weights, _, _ = training_utils.handle_partial_sample_weights(y, sample_weights, sample_weight_modes, check_all_flat=True)
        inputs = pack_x_y_sample_weight(x, y, sample_weights)
        num_samples = set((int(i.shape[0]) for i in nest.flatten(inputs))).pop()
        _check_data_cardinality(inputs)
        if not batch_size:
            batch_size = int(math.ceil(num_samples / steps)) if steps else 32
        self._size = int(math.ceil(num_samples / batch_size))
        self._batch_size = batch_size
        num_full_batches = int(num_samples // batch_size)
        self._partial_batch_size = num_samples % batch_size
        if isinstance(shuffle, str):
            shuffle = shuffle.lower()
        self._shuffle = shuffle
        indices_dataset = dataset_ops.DatasetV2.range(1)
        if shuffle != 'batch':
            indices_dataset = indices_dataset.repeat(epochs)

        def permutation(_):
            indices = math_ops.range(num_samples, dtype=dtypes.int64)
            if shuffle and shuffle != 'batch':
                indices = random_ops.random_shuffle(indices)
            return indices
        indices_dataset = indices_dataset.map(permutation).prefetch(1)

        def slice_batch_indices(indices):
            """Convert a Tensor of indices into a dataset of batched indices.

      This step can be accomplished in several ways. The most natural is to
      slice the Tensor in a Dataset map. (With a condition on the upper index to
      handle the partial batch.) However it turns out that coercing the Tensor
      into a shape which is divisible by the batch size (and handling the last
      partial batch separately) allows for a much more favorable memory access
      pattern and improved performance.

      Args:
        indices: Tensor which determines the data order for an entire epoch.

      Returns:
        A Dataset of batched indices.
      """
            num_in_full_batch = num_full_batches * batch_size
            first_k_indices = array_ops.slice(indices, [0], [num_in_full_batch])
            first_k_indices = array_ops.reshape(first_k_indices, [num_full_batches, batch_size])
            flat_dataset = dataset_ops.DatasetV2.from_tensor_slices(first_k_indices)
            if self._partial_batch_size:
                index_remainder = dataset_ops.DatasetV2.from_tensors(array_ops.slice(indices, [num_in_full_batch], [self._partial_batch_size]))
                flat_dataset = flat_dataset.concatenate(index_remainder)
            if shuffle == 'batch':
                flat_dataset = flat_dataset.shuffle(1024).repeat(epochs)
            return flat_dataset
        indices_dataset = indices_dataset.flat_map(slice_batch_indices)
        dataset = self.slice_inputs(indices_dataset, inputs)
        if shuffle == 'batch':

            def shuffle_batch(*batch):
                return nest.map_structure(random_ops.random_shuffle, batch)
            dataset = dataset.map(shuffle_batch)
        self._dataset = dataset

    def slice_inputs(self, indices_dataset, inputs):
        """Slice inputs into a Dataset of batches.

    Given a Dataset of batch indices and the unsliced inputs,
    this step slices the inputs in a parallelized fashion
    and produces a dataset of input batches.

    Args:
      indices_dataset: A Dataset of batched indices
      inputs: A python data structure that contains the inputs, targets,
        and possibly sample weights.

    Returns:
      A Dataset of input batches matching the batch indices.
    """
        dataset = dataset_ops.DatasetV2.zip((indices_dataset, dataset_ops.DatasetV2.from_tensors(inputs).repeat()))

        def grab_batch(i, data):
            return nest.map_structure(lambda d: array_ops.gather(d, i, axis=0), data)
        dataset = dataset.map(grab_batch, num_parallel_calls=dataset_ops.AUTOTUNE)
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        if self._shuffle:
            options.experimental_external_state_policy = options_lib.ExternalStatePolicy.IGNORE
        dataset = dataset.with_options(options)
        return dataset

    def get_dataset(self):
        return self._dataset

    def get_size(self):
        return self._size

    def batch_size(self):
        return self._batch_size

    def has_partial_batch(self):
        return self._partial_batch_size > 0

    def partial_batch_size(self):
        return self._partial_batch_size or None

    def should_recreate_iterator(self):
        return False