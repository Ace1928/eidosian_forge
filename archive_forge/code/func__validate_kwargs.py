import math
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
def _validate_kwargs(cls_name, kwargs, support_partition=True):
    for kwarg in kwargs:
        if kwarg not in [_PARTITION_SHAPE, _PARTITION_OFFSET]:
            raise TypeError('Unknown keyword arguments: %s' % kwarg)
        elif not support_partition:
            raise ValueError("%s initializer doesn't support partition-related arguments" % cls_name)