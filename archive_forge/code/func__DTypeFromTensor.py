from tensorflow.core.config import flags
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
def _DTypeFromTensor(tensor):
    """Extract either `tensor.dtype` or the unanimous sub-type of a variant."""
    dtype = tensor.dtype
    if dtype.base_dtype == dtypes.variant:
        if isinstance(tensor, ops.EagerTensor):
            handle_data = tensor._handle_data
        else:
            handle_data = handle_data_util.get_resource_handle_data(tensor)
        if handle_data is not None and handle_data.is_set and handle_data.shape_and_type:
            first_type = handle_data.shape_and_type[0].dtype
            if first_type != types_pb2.DT_INVALID and all((shape_and_type.dtype == first_type for shape_and_type in handle_data.shape_and_type)):
                return first_type
    return dtype