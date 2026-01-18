from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _infer_state_dtype(explicit_dtype, state):
    """Infer the dtype of an RNN state.

  Args:
    explicit_dtype: explicitly declared dtype or None.
    state: RNN's hidden state. Must be a Tensor or a nested iterable containing
      Tensors.

  Returns:
    dtype: inferred dtype of hidden state.

  Raises:
    ValueError: if `state` has heterogeneous dtypes or is empty.
  """
    if explicit_dtype is not None:
        return explicit_dtype
    elif nest.is_nested(state):
        inferred_dtypes = [element.dtype for element in nest.flatten(state)]
        if not inferred_dtypes:
            raise ValueError(f'Unable to infer dtype from argument state={state}.')
        all_same = all((x == inferred_dtypes[0] for x in inferred_dtypes))
        if not all_same:
            raise ValueError(f'Argument state={state} has tensors of different inferred dtypes. Unable to infer a single representative dtype. Dtypes received: {inferred_dtypes}')
        return inferred_dtypes[0]
    else:
        return state.dtype