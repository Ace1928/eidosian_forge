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
class ListsOfScalarsDataAdapter(DataAdapter):
    """Adapter that handles lists of scalars and lists of lists of scalars."""

    @staticmethod
    def can_handle(x, y=None):
        handles_x = ListsOfScalarsDataAdapter._is_list_of_scalars(x)
        handles_y = True
        if y is not None:
            handles_y = ListsOfScalarsDataAdapter._is_list_of_scalars(y)
        return handles_x and handles_y

    @staticmethod
    def _is_list_of_scalars(inp):
        if isinstance(inp, (float, int, str, bytes, bytearray)):
            return True
        if isinstance(inp, (list, tuple)) and inp:
            return ListsOfScalarsDataAdapter._is_list_of_scalars(inp[0])
        return False

    def __init__(self, x, y=None, sample_weights=None, sample_weight_modes=None, batch_size=None, shuffle=False, **kwargs):
        super(ListsOfScalarsDataAdapter, self).__init__(x, y, **kwargs)
        x = np.asarray(x)
        if y is not None:
            y = np.asarray(y)
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights)
        sample_weight_modes = broadcast_sample_weight_modes(sample_weights, sample_weight_modes)
        self._internal_adapter = TensorLikeDataAdapter(x, y=y, sample_weights=sample_weights, sample_weight_modes=sample_weight_modes, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def get_dataset(self):
        return self._internal_adapter.get_dataset()

    def get_size(self):
        return self._internal_adapter.get_size()

    def batch_size(self):
        return self._internal_adapter.batch_size()

    def has_partial_batch(self):
        return self._internal_adapter.has_partial_batch()

    def partial_batch_size(self):
        return self._internal_adapter.partial_batch_size()

    def should_recreate_iterator(self):
        return True