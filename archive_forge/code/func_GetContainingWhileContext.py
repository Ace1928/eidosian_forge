from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import variables
def GetContainingWhileContext(ctxt, stop_ctxt=None):
    """Returns the first ancestor WhileContext of `ctxt`.

  Returns `ctxt` if `ctxt` is a WhileContext, or None if `ctxt` is not in a
  while loop.

  Args:
    ctxt: ControlFlowContext
    stop_ctxt: ControlFlowContext, optional. If provided, the search will end
      if it sees stop_ctxt.

  Returns:
    `ctxt` if `ctxt` is a WhileContext, the most nested WhileContext containing
    `ctxt`, or None if `ctxt` is not in a while loop.  If `stop_ctxt` is not
    `None`, this returns `ctxt` if it matches `stop_ctxt` in its traversal.
  """
    while ctxt:
        if ctxt.IsWhileContext() or ctxt == stop_ctxt:
            return ctxt
        ctxt = ctxt.outer_context
    return None