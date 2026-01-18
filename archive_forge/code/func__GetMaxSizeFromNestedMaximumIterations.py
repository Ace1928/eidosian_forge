from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
def _GetMaxSizeFromNestedMaximumIterations(value, while_ctxt):
    """Calculate a max_size for use by stack ops inside an XLA while_loop.

  Args:
    value: The value inside the while_loop forward context.  Used for printing
      error messages.
    while_ctxt: The forward context inside which value resides.  This does not
      always match the value's immediate context, as `value` may be inside e.g.
      a cond context inside the while_loop.

  Returns:
    A tensor containing the `max_size` to feed to a Stack initializer.

  Raises:
    ValueError: If `value` is nested inside a `while_loop` that either
      lacks a `maximum_iterations` parameter, or the `maximum_iterations`
      parameter:

        - is inside a `while_loop` that is a parent of the calling context, and
        - cannot be evaluated at graph build time to a constant.
  """
    value_name = value.name
    curr_ctxt = ops.get_default_graph()._get_control_flow_context()
    curr_ctxt_name = curr_ctxt.name if curr_ctxt is not None else ''
    max_size = constant_op.constant(1)
    while while_ctxt not in (None, curr_ctxt):
        max_iter = while_ctxt.maximum_iterations
        if max_iter is None:
            raise ValueError("Cannot create a gradient accumulator for tensor '%s' inside XLA while_loop because maximum_iterations was not passed to the tf.while_loop call ('%s')." % (value_name, while_ctxt.name))
        max_iter_ctxt = max_iter.op._get_control_flow_context()
        if util.IsContainingContext(curr_ctxt, max_iter_ctxt):
            max_size *= max_iter
        else:
            const_max_iter = tensor_util.constant_value(max_iter)
            if const_max_iter is None:
                raise ValueError("Cannot create a gradient accumulator for tensor '%s' inside XLA while_loop. maximum_iterations tensor '%s' for while_loop context '%s' must be statically known (e.g. a constant value or known shape dimension), or be defined at or outside the while loop context '%s' (currently defined in '%s')." % (value_name, max_iter.name, while_ctxt.name, curr_ctxt_name, max_iter_ctxt.name))
            max_size *= const_max_iter
        while_ctxt = util.GetContainingWhileContext(while_ctxt.outer_context, stop_ctxt=curr_ctxt)
    return max_size