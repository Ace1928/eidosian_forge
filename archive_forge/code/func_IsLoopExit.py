import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def IsLoopExit(op):
    """Return true if `op` is an Exit."""
    return op.type == 'Exit' or op.type == 'RefExit'