import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def IsLoopMerge(op):
    """Return true if `op` is the Merge for a while loop."""
    if IsMerge(op):
        ctxt = op._get_control_flow_context()
        return ctxt is not None and ctxt.IsWhileContext() and (not IsCondMerge(op))
    return False