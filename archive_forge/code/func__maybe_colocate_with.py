import contextlib
import traceback
import weakref
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import trace
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export
@contextlib.contextmanager
def _maybe_colocate_with(self, value):
    """Colocate operations with an internal colocation group or `value`.

    Args:
      value: `Tensor`, the tensor to try to colocate with.

    Yields:
      Does not yield anything, but the new context is a colocation context.

    If no internal colocation group is set, colocate with `value` and set
    the internal colocation group to be value.
    """
    if not self._colocate_with_first_write_call:
        yield
    else:
        if not self._colocate_with:
            self._colocate_with.append(value)
        with ops.colocate_with(self._colocate_with[0]):
            yield