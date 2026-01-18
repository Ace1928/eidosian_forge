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
class OutputsAggregator(Aggregator):
    """Aggregator that concatenates outputs."""
    _structure = None

    def create(self, batch_outs):
        self._structure = nest.get_traverse_shallow_structure(lambda x: not is_composite_or_composite_value(x), batch_outs)
        batch_outs = nest.flatten_up_to(self._structure, batch_outs)
        for batch_element in batch_outs:
            if is_composite_or_composite_value(batch_element):
                self.results.append(ConcatAggregator(self.batch_size))
            elif isinstance(batch_element, np.ndarray):
                self.results.append(ConcatAggregator(self.batch_size) if self.use_steps else SliceAggregator(self.num_samples, self.batch_size))
            else:
                raise RuntimeError('Attempted to aggregate unsupported object {}.'.format(batch_element))
            self.results[-1].create(batch_element)

    def aggregate(self, batch_outs, batch_start=None, batch_end=None):
        batch_outs = nest.flatten_up_to(self._structure, batch_outs)
        for batch_element, result in zip(batch_outs, self.results):
            result.aggregate(batch_element, batch_start, batch_end)

    def finalize(self):
        for result in self.results:
            result.finalize()
        self.results = [i.results for i in self.results]
        self.results = nest.pack_sequence_as(self._structure, self.results)