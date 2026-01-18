from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_shape
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect
def _op_is_in_tf_version(op, version):
    if version == 1:
        return tf_export.get_v1_names(tf_decorator.unwrap(op)[1]) or op in _V2_OPS_THAT_ARE_DELEGATED_TO_FROM_V1_OPS
    elif version == 2:
        return tf_export.get_v2_names(tf_decorator.unwrap(op)[1])
    else:
        raise ValueError('Expected version 1 or 2.')