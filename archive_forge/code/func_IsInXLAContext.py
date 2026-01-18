import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def IsInXLAContext(op):
    try:
        xla_compile = op.get_attr('_XlaCompile')
        if xla_compile:
            return True
    except ValueError:
        pass
    ctxt = op._get_control_flow_context()
    return GetContainingXLAContext(ctxt) is not None