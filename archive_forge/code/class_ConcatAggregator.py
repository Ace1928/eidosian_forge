import abc
import atexit
import collections
import functools
import multiprocessing.pool
import threading
import time
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
class ConcatAggregator(Aggregator):
    """Combine tensor-likes which cannot be merged on the fly.

  This class expects to aggregate a single tensor-like rather than a nested
  structure of tensor-likes.
  """

    def __init__(self, batch_size):
        self.composite = None
        super(ConcatAggregator, self).__init__(use_steps=True, num_samples=None, steps=None, batch_size=batch_size)

    def create(self, batch_element):
        self.composite = is_composite_or_composite_value(batch_element)

    def aggregate(self, batch_element, batch_start=None, batch_end=None):
        if self.batch_size and self.batch_size < batch_element.shape[0]:
            raise ValueError('Mismatch between expected batch size and model output batch size. Output shape = {}, expected output shape = shape {}'.format(batch_element.shape, (self.batch_size,) + batch_element.shape[1:]))
        self.results.append(batch_element)

    def finalize(self):
        if len(self.results) == 1:
            self.results = self.results[0]
        elif self.composite:
            results = self.results[0]
            for r in self.results[1:]:
                results = _append_composite_tensor(results, r)
            self.results = results
        else:
            self.results = np.concatenate(self.results, axis=0)