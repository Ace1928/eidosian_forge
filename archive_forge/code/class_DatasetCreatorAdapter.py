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
class DatasetCreatorAdapter(DataAdapter):
    """Adapter that handles dataset functions."""

    def __init__(self, x, y, steps=None, distribution_strategy=None, **kwargs):
        super(DatasetCreatorAdapter, self).__init__(x, **kwargs)
        if not isinstance(x, dataset_creator.DatasetCreator):
            raise TypeError('The input of a `DatasetCreatorAdapter` should be a `DatasetCreator` but it received type {}.'.format(type(x)))
        if steps is None:
            raise ValueError('When using a `tf.keras.utils.experimental.DatasetCreator`, `steps_per_epoch`, `validation_steps` or `steps` argument must be provided in `Model.fit`, `Model.evaluate`, or `Model.predict`.')
        self.dataset_creator = x
        self.steps = steps
        self.strategy = distribution_strategy

    @staticmethod
    def can_handle(x, y=None):
        if isinstance(x, dataset_creator.DatasetCreator):
            assert y is None
            return True

    def should_recreate_iterator(self):
        return False

    def get_size(self):
        return None

    def get_dataset(self):
        return self.strategy.distribute_datasets_from_function(self.dataset_creator, options=self.dataset_creator.input_options)

    def batch_size(self):
        raise NotImplementedError()

    def has_partial_batch(self):
        raise NotImplementedError()

    def partial_batch_size(self):
        raise NotImplementedError()