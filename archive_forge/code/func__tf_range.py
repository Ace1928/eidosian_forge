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
def _tf_range(start_or_stop, stop, step):
    """Overload of range_ that generates a TF range tensor."""
    if step is not UNSPECIFIED:
        return math_ops.range(start_or_stop, stop, step)
    if stop is not UNSPECIFIED:
        stop = math_ops.maximum(start_or_stop, stop)
        return math_ops.range(start_or_stop, stop)
    start_or_stop = math_ops.maximum(start_or_stop, 0)
    return math_ops.range(start_or_stop)