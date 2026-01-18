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
def _process_tensorlike(inputs):
    """Process tensor-like inputs.

  This function:

  (1) Converts `Numpy` arrays to `Tensor`s.
  (2) Converts `Scipy` sparse matrices to `SparseTensor`s.
  (2) Converts `list`s to `tuple`s (for `tf.data` support).

  Args:
    inputs: Structure of `Tensor`s, `NumPy` arrays, or tensor-like.

  Returns:
    Structure of `Tensor`s or tensor-like.
  """

    def _convert_numpy_and_scipy(x):
        if isinstance(x, np.ndarray):
            dtype = None
            if issubclass(x.dtype.type, np.floating):
                dtype = backend.floatx()
            return tensor_conversion.convert_to_tensor_v2_with_dispatch(x, dtype=dtype)
        elif _is_scipy_sparse(x):
            return _scipy_sparse_to_sparse_tensor(x)
        return x
    inputs = nest.map_structure(_convert_numpy_and_scipy, inputs)
    return nest.list_to_tuple(inputs)