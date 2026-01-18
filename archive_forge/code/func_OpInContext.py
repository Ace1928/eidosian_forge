import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def OpInContext(op, ctxt):
    return IsContainingContext(op._get_control_flow_context(), ctxt)