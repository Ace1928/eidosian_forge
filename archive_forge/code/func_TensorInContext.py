import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def TensorInContext(tensor, ctxt):
    return OpInContext(tensor.op, ctxt)