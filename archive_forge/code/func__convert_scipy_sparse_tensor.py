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
def _convert_scipy_sparse_tensor(value, expected_input):
    """Handle scipy sparse tensor conversions.

  This method takes a value 'value' and returns the proper conversion. If
  value is a scipy sparse tensor and the expected input is a dense tensor,
  we densify 'value'. If value is a scipy sparse tensor and the expected input
  is a TF SparseTensor, we convert 'value' to a SparseTensor. If 'value' is
  not a scipy sparse tensor, or scipy is not imported, we pass it through
  unchanged.

  Args:
    value: An object that may be a scipy sparse tensor
    expected_input: The expected input placeholder.

  Returns:
    The possibly-converted 'value'.
  """
    if issparse is not None and issparse(value):
        if backend.is_sparse(expected_input):
            sparse_coo = value.tocoo()
            row, col = (sparse_coo.row, sparse_coo.col)
            data, shape = (sparse_coo.data, sparse_coo.shape)
            indices = np.concatenate((np.expand_dims(row, 1), np.expand_dims(col, 1)), 1)
            return sparse_tensor.SparseTensor(indices, data, shape)
        else:
            if ops.executing_eagerly_outside_functions():
                raise ValueError('A SciPy sparse matrix was passed to a model that expects dense inputs. Please densify your inputs first, such as by calling `x.toarray().')
            return value.toarray()
    else:
        return value