from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_config
from tensorflow.python.ops.gen_parsing_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _parse_single_sequence_example_raw(serialized, context, feature_list, debug_name, name=None):
    """Parses a single `SequenceExample` proto.

  Args:
    serialized: A scalar (0-D Tensor) of type string, a single binary serialized
      `SequenceExample` proto.
    context: A `ParseOpParams` containing the parameters for the parse op for
      the context features.
    feature_list: A `ParseOpParams` containing the parameters for the parse op
      for the feature_list features.
    debug_name: A scalar (0-D Tensor) of strings (optional), the name of the
      serialized proto.
    name: A name for this operation (optional).

  Returns:
    A tuple of two `dict`s, each mapping keys to `Tensor`s and `SparseTensor`s.
    The first dict contains the context key/values.
    The second dict contains the feature_list key/values.

  Raises:
    TypeError: if feature_list.dense_defaults is not either None or a dict.
  """
    with ops.name_scope(name, 'ParseSingleExample', [serialized, debug_name]):
        serialized = ops.convert_to_tensor(serialized, name='serialized')
        serialized = _assert_scalar(serialized, 'serialized')
    return _parse_sequence_example_raw(serialized, debug_name, context, feature_list, name)[:2]