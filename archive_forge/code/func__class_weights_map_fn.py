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
def _class_weights_map_fn(*data):
    """Convert `class_weight` to `sample_weight`."""
    x, y, sw = unpack_x_y_sample_weight(data)
    if nest.is_nested(y):
        raise ValueError('`class_weight` is only supported for Models with a single output.')
    if y.shape.rank > 2:
        raise ValueError('`class_weight` not supported for 3+ dimensional targets.')
    y_classes = smart_cond.smart_cond(y.shape.rank == 2 and backend.shape(y)[1] > 1, lambda: backend.argmax(y, axis=1), lambda: math_ops.cast(backend.reshape(y, (-1,)), dtypes.int64))
    cw = array_ops.gather_v2(class_weight_tensor, y_classes)
    if sw is not None:
        cw = math_ops.cast(cw, sw.dtype)
        sw, cw = expand_1d((sw, cw))
        sw = sw * cw
    else:
        sw = cw
    return (x, y, sw)