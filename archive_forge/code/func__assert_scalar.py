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
def _assert_scalar(value, name):
    """Asserts that `value` is scalar, and returns `value`."""
    value_rank = value.shape.rank
    if value_rank is None:
        check = control_flow_assert.Assert(math_ops.equal(array_ops.rank(value), 0), ['Input %s must be a scalar' % name], name='%sIsScalar' % name.capitalize())
        result = control_flow_ops.with_dependencies([check], value, name='%sDependencies' % name)
        result.set_shape([])
        return result
    elif value_rank == 0:
        return value
    else:
        raise ValueError('Input %s must be a scalar' % name)