import inspect
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
def _py_sorted(iterable, key, reverse):
    if key is not UNSPECIFIED and reverse is UNSPECIFIED:
        return sorted(iterable, key=key)
    if key is UNSPECIFIED and reverse is not UNSPECIFIED:
        return sorted(iterable, reverse=reverse)
    if key is not UNSPECIFIED and reverse is not UNSPECIFIED:
        return sorted(iterable, key=key, reverse=reverse)
    return sorted(iterable)