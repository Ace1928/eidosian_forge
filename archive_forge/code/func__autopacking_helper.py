import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def _autopacking_helper(list_or_tuple, dtype, name):
    """Converts the given list or tuple to a tensor by packing.

  Args:
    list_or_tuple: A (possibly nested) list or tuple containing a tensor.
    dtype: The element type of the returned tensor.
    name: A name for the returned tensor.

  Returns:
    A `tf.Tensor` with value equivalent to `list_or_tuple`.
  """
    if context.executing_eagerly():
        if all((isinstance(elem, core.Tensor) for elem in list_or_tuple)):
            return gen_array_ops.pack(list_or_tuple, name=name)
    must_pack = False
    converted_elems = []
    with ops.name_scope(name) as scope:
        for i, elem in enumerate(list_or_tuple):
            if isinstance(elem, core.Tensor):
                if dtype is not None and elem.dtype.base_dtype != dtype:
                    raise TypeError(f'Cannot convert a list containing a tensor of dtype {elem.dtype} to {dtype} (Tensor is: {elem!r})')
                converted_elems.append(elem)
                must_pack = True
            elif isinstance(elem, (list, tuple)):
                converted_elem = _autopacking_helper(elem, dtype, str(i))
                if isinstance(converted_elem, core.Tensor):
                    must_pack = True
                converted_elems.append(converted_elem)
            else:
                converted_elems.append(elem)
        if must_pack:
            elems_as_tensors = []
            for i, elem in enumerate(converted_elems):
                if isinstance(elem, core.Tensor):
                    elems_as_tensors.append(elem)
                else:
                    elems_as_tensors.append(constant_op.constant(elem, dtype=dtype, name=str(i)))
            return gen_array_ops.pack(elems_as_tensors, name=scope)
        else:
            return converted_elems