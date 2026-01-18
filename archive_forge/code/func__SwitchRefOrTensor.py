import abc
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.gen_control_flow_ops import *
from tensorflow.python.util import compat
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _SwitchRefOrTensor(data, pred, name='Switch'):
    """Forwards `data` to an output determined by `pred`.

  If `pred` is false, the `data` input is forwarded to the first output.
  Otherwise, the data goes to the second output.

  This op handles `Tensor`s and `IndexedSlices`.

  Args:
    data: The tensor to be forwarded to the appropriate output.
    pred: A scalar that specifies which output port will receive data.
    name: A name for this operation (optional).

  Returns:
    `(output_false, output_true)`: If `pred` is true, data will be forwarded to
    `output_true`, otherwise it goes to `output_false`.

  Raises:
    TypeError: if data is not a Tensor or IndexedSlices
  """
    data = ops.convert_to_tensor_or_composite(data, name='data')
    with ops.colocate_with(data, ignore_existing=True):
        if isinstance(data, tensor_lib.Tensor):
            if data.dtype._is_ref_dtype:
                return ref_switch(data, pred, name=name)
        return switch(data, pred, name=name)