import functools
import numbers
import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops.gen_nn_ops import *
from tensorflow.python.platform import device_context
from tensorflow.python.platform import build_info
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup
from tensorflow.python.util.tf_export import tf_export
def _get_sequence(value, n, channel_index, name):
    """Formats a value input for gen_nn_ops."""
    if value is None:
        return [1] * (n + 2)
    if isinstance(value, list):
        pass
    elif isinstance(value, tuple):
        value = list(value)
    elif isinstance(value, int):
        value = [value]
    elif not isinstance(value, collections_abc.Sized):
        value = [value]
    else:
        value = list(value)
    len_value = len(value)
    if len_value == n + 2:
        return value
    if len_value == 1:
        value = value * n
    elif len_value != n:
        raise ValueError(f'{name} should be of length 1, {n} or {n + 2}. Received: {name}={value} of length {len_value}')
    if channel_index == 1:
        return [1, 1] + value
    else:
        return [1] + value + [1]