import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
def _eager_fill(dims, value, ctx):
    """Eager-only version of Fill op; requires value is an eager Tensor."""
    attr_t = value.dtype.as_datatype_enum
    dims = convert_to_eager_tensor(dims, ctx, dtypes.int32)
    inputs_flat = [dims, value]
    attrs = ('T', attr_t, 'index_type', types_pb2.DT_INT32)
    [result] = execute.execute(b'Fill', 1, inputs=inputs_flat, attrs=attrs, ctx=ctx)
    return result