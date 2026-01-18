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
def _scipy_sparse_to_sparse_tensor(t):
    """Converts a SciPy sparse matrix to a SparseTensor."""
    sparse_coo = t.tocoo()
    row, col = (sparse_coo.row, sparse_coo.col)
    data, shape = (sparse_coo.data, sparse_coo.shape)
    if issubclass(data.dtype.type, np.floating):
        data = data.astype(backend.floatx())
    indices = np.concatenate((np.expand_dims(row, axis=1), np.expand_dims(col, axis=1)), axis=1)
    return sparse_tensor.SparseTensor(indices, data, shape)