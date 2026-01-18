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
class _ClusterCoordinatorDataHandler(DataHandler):
    """A `DataHandler` that is compatible with `ClusterCoordinator`."""

    def __init__(self, x, y=None, **kwargs):
        if not isinstance(x, dataset_creator.DatasetCreator):
            x = self._convert_to_dataset_creator(x, y, **kwargs)
        super().__init__(x=x, **kwargs)

    def _convert_to_dataset_creator(self, x, y, **kwargs):
        """Converts non-tf.data.Dataset to `DatasetCreator` instances."""

        def _dataset_fn(input_context):
            del input_context
            data_adapter_cls = select_data_adapter(x, y)
            return data_adapter_cls(x=x, y=y, **kwargs).get_dataset()
        if isinstance(x, _get_tensor_types()) and isinstance(y, _get_tensor_types()):
            return dataset_creator.DatasetCreator(_dataset_fn)
        else:
            raise NotImplementedError('Only `tf.keras.utils.experimental.DatasetCreator`, `tf.Tensor`, numpy arrays and pandas dataframes are supported types at this time.')

    def _configure_dataset_and_inferred_steps(self, strategy, x, steps_per_epoch, class_weight, distribute):
        if not isinstance(x, dataset_creator.DatasetCreator):
            raise TypeError('When using `ParameterServerStrategy`, `x` must be a `DatasetCreator`.')

        def per_worker_dataset_fn():
            return strategy.distribute_datasets_from_function(x, options=x.input_options)
        self._dataset = self._model._cluster_coordinator.create_per_worker_dataset(per_worker_dataset_fn)
        if steps_per_epoch == -1:
            self._inferred_steps = None
            self._log_indefinite_training_warning()
        else:
            self._inferred_steps = steps_per_epoch

    def sync(self):
        self._model._cluster_coordinator.join()