import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _check_enqueue_dtypes(self, vals):
    """Validate and convert `vals` to a list of `Tensor`s.

    The `vals` argument can be a Tensor, a list or tuple of tensors, or a
    dictionary with tensor values.

    If it is a dictionary, the queue must have been constructed with a
    `names` attribute and the dictionary keys must match the queue names.
    If the queue was constructed with a `names` attribute, `vals` must
    be a dictionary.

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary..

    Returns:
      A list of `Tensor` objects.

    Raises:
      ValueError: If `vals` is invalid.
    """
    if isinstance(vals, dict):
        if not self._names:
            raise ValueError('Queue must have names to enqueue a dictionary')
        if sorted(self._names, key=str) != sorted(vals.keys(), key=str):
            raise ValueError(f'Keys in dictionary to enqueue do not match names of Queue.  Dictionary: {sorted(vals.keys())},Queue: {sorted(self._names)}')
        vals = [vals[k] for k in self._names]
    else:
        if self._names:
            raise ValueError('You must enqueue a dictionary in a Queue with names')
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
    tensors = []
    for i, (val, dtype) in enumerate(zip(vals, self._dtypes)):
        tensors.append(ops.convert_to_tensor(val, dtype=dtype, name='component_%d' % i))
    return tensors