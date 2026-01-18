import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def GetLoopConstantEnter(value):
    """Return the enter op if we can infer `value` to be a loop invariant."""
    id_ops = {'Switch', 'RefSwitch', 'Identity', 'RefIdentity'}
    op = value.op
    while op.type in id_ops:
        op = op.inputs[0].op
    return op if IsLoopConstantEnter(op) else None