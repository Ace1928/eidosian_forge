import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def CheckInputFromValidContext(op, input_op):
    """Returns whether `input_op` can be used from `op`s context.

  Conceptually, only inputs from op's while context or any ancestor while
  context (including outside of any context) are valid. In practice, there are
  many other edge cases as well.

  Args:
    op: Operation
    input_op: Operation

  Raises:
    ValueError: if input_op is from an invalid context.
  """
    op_ctxt = op._get_control_flow_context()
    input_ctxt = GetOutputContext(input_op)
    valid = False
    if not input_ctxt:
        valid = True
    elif op_ctxt is input_ctxt:
        valid = True
    else:
        while_ctxt = GetContainingWhileContext(op_ctxt)
        input_while_ctxt = GetContainingWhileContext(input_ctxt)
        if while_ctxt is None:
            if input_while_ctxt is None:
                valid = True
            if IsLoopEnter(op):
                valid = True
            if IsSwitch(op):
                valid = True
        elif IsContainingContext(while_ctxt, input_while_ctxt):
            valid = True
        elif while_ctxt.grad_state and IsContainingContext(while_ctxt.grad_state.forward_context, input_while_ctxt):
            valid = True
        elif while_ctxt.grad_state and while_ctxt.grad_state.forward_context is input_while_ctxt._outer_context:
            valid = True
        elif input_while_ctxt.grad_state and input_while_ctxt.grad_state.forward_context is while_ctxt:
            valid = True
        elif input_while_ctxt.grad_state and input_ctxt.grad_state.forward_context.grad_state and (input_ctxt.grad_state.forward_context.grad_state.forward_context is while_ctxt):
            valid = True
    if not valid:
        if while_ctxt:
            error_msg = f"Cannot use '{input_op.name}' as input to '{op.name}' because they are in different while loops."
        else:
            error_msg = f"Cannot use '{input_op.name}' as input to '{op.name}' because '{input_op.name}' is in a while loop."
        log_msg = error_msg
        log_msg += '\n\n%s while context: %s' % (op.name, while_ctxt)
        log_msg += '\n%s while context: %s' % (input_op.name, input_while_ctxt)
        log_msg += '\n\nTraceback for %s:\n%s\nTraceback for %s:\n%s\n' % (op.name, ''.join(traceback.format_list(op.traceback)), input_op.name, ''.join(traceback.format_list(input_op.traceback)))
        logging.info(log_msg)
        raise ValueError(error_msg + ' See info log for more details.')