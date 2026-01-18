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
class Aggregator(object, metaclass=abc.ABCMeta):
    """Abstract base class used to aggregate batch-level outputs of a loop.

  Attributes:
    use_steps: Whether the loop is using `step` or `batch_size`.
    num_samples: Total number of samples: `batch_size * num_batches`.
    steps: Total number of steps.
    batch_size: Batch size. It is used for validation checks between inputs and
      outputs.
    results: What to return at the end of the aggregation loop.
  """

    def __init__(self, use_steps, num_samples=None, steps=None, batch_size=None):
        self.use_steps = use_steps
        self.num_samples = num_samples
        self.steps = steps
        self.batch_size = batch_size
        self.results = []

    @abc.abstractmethod
    def create(self, batch_outs):
        """Creates the initial results from the first batch outputs.

    Args:
      batch_outs: A list of batch-level outputs.
    """
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def aggregate(self, batch_outs, batch_start=None, batch_end=None):
        """Aggregates batch-level results into total results.

    Args:
      batch_outs: A list of batch-level outputs.
      batch_start: The start index of this batch. Always `None` if `use_steps`
        is `True`.
      batch_end: The end index of this batch. Always `None` if `use_steps` is
        `True`.
    """
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def finalize(self):
        """Prepares the total results to be returned."""
        raise NotImplementedError('Must be implemented in subclasses.')