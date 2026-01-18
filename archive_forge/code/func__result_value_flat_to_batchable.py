import re
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _result_value_flat_to_batchable(result_value_flat, result_flat_signature):
    """Converts result_value_flat -> result_value_batchable."""
    result_value_batchable = []
    for r_value, r_spec in zip(result_value_flat, result_flat_signature):
        if isinstance(r_spec, tensor_spec.TensorSpec):
            result_value_batchable.append(r_value)
        else:
            if not r_spec.is_compatible_with(r_value):
                raise ValueError('Error in map_fn:\n  Expected `fn` to return a:\n    %s\n  But it returned a:\n    %s\n    (value=%s)\n  To fix, update the `fn_output_signature` (or `dtype`) argument to `map_fn`.' % (r_spec, type_spec.type_spec_from_value(r_value), r_value))
            result_value_batchable.extend(r_spec._to_tensor_list(r_value))
    return result_value_batchable