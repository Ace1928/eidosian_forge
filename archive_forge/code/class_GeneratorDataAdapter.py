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
class GeneratorDataAdapter(DataAdapter):
    """Adapter that handles python generators and iterators."""

    @staticmethod
    def can_handle(x, y=None):
        return (hasattr(x, '__next__') or hasattr(x, 'next')) and hasattr(x, '__iter__') and (not isinstance(x, data_utils.Sequence))

    def __init__(self, x, y=None, sample_weights=None, workers=1, use_multiprocessing=False, max_queue_size=10, model=None, **kwargs):
        kwargs.pop('shuffle', None)
        if not is_none_or_empty(y):
            raise ValueError('`y` argument is not supported when using python generator as input.')
        if not is_none_or_empty(sample_weights):
            raise ValueError('`sample_weight` argument is not supported when using python generator as input.')
        super(GeneratorDataAdapter, self).__init__(x, y, **kwargs)
        peek, x = self._peek_and_restore(x)
        peek = self._standardize_batch(peek)
        peek = _process_tensorlike(peek)
        if model is not None and (not model.built):
            concrete_x, _, _ = unpack_x_y_sample_weight(peek)
            model.distribute_strategy.run(lambda x: model(x, training=False), args=(concrete_x,))
        self._first_batch_size = int(nest.flatten(peek)[0].shape[0])

        def _get_dynamic_shape(t):
            shape = t.shape
            if shape.rank is None:
                return shape
            return tensor_shape.TensorShape([None for _ in shape.as_list()])
        output_shapes = nest.map_structure(_get_dynamic_shape, peek)
        output_types = nest.map_structure(lambda t: t.dtype, peek)
        generator_fn = self._handle_multiprocessing(x, workers, use_multiprocessing, max_queue_size)

        def wrapped_generator():
            for data in generator_fn():
                yield self._standardize_batch(data)
        dataset = dataset_ops.DatasetV2.from_generator(wrapped_generator, output_types, output_shapes=output_shapes)
        if workers == 1 and (not use_multiprocessing):
            dataset = dataset.prefetch(1)
        self._dataset = dataset

    def _standardize_batch(self, data):
        """Standardizes a batch output by a generator."""
        x, y, sample_weight = unpack_x_y_sample_weight(data)
        data = pack_x_y_sample_weight(x, y, sample_weight)
        data = nest.list_to_tuple(data)

        def _convert_dtype(t):
            if isinstance(t, np.ndarray) and issubclass(t.dtype.type, np.floating):
                return np.array(t, dtype=backend.floatx())
            return t
        data = nest.map_structure(_convert_dtype, data)
        return data

    @staticmethod
    def _peek_and_restore(x):
        peek = next(x)
        return (peek, itertools.chain([peek], x))

    def _handle_multiprocessing(self, x, workers, use_multiprocessing, max_queue_size):
        """Create a callable, possibly including an Enqueuer."""
        if workers > 1 or (workers > 0 and use_multiprocessing):

            def generator_fn():
                enqueuer = data_utils.GeneratorEnqueuer(x, use_multiprocessing=use_multiprocessing)
                enqueuer.start(workers=workers, max_queue_size=max_queue_size)
                return enqueuer.get()
        else:
            generator_fn = lambda: x
        return generator_fn

    def get_dataset(self):
        return self._dataset

    def get_size(self):
        return None

    def batch_size(self):
        return None

    def representative_batch_size(self):
        return self._first_batch_size

    def has_partial_batch(self):
        return False

    def partial_batch_size(self):
        return

    def should_recreate_iterator(self):
        return False