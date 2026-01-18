import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def IsCondSwitch(op):
    """Return true if `op` is the Switch for a conditional."""
    if not IsSwitch(op):
        return False
    if not op.outputs:
        return False
    is_cond_switch = True
    for o in op.outputs:
        for c in o.consumers():
            ctxt = c._get_control_flow_context()
            if IsLoopEnter(c):
                ctxt = ctxt.outer_context
            is_cond_switch = is_cond_switch and (ctxt is not None and ctxt.IsCondContext())
    return is_cond_switch