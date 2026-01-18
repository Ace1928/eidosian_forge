import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def IsCondMerge(op):
    """Return true if `op` is the Merge for a conditional."""
    if not IsMerge(op):
        return False
    if not op.inputs:
        return False
    is_cond_merge = True
    for i in op.inputs:
        ctxt = GetOutputContext(i.op)
        is_cond_merge = is_cond_merge and ctxt is not None and ctxt.IsCondContext()
    return is_cond_merge