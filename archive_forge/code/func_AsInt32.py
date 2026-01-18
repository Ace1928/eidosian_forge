from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
def AsInt32(x):
    return x if op.inputs[0].dtype == dtypes.int32 else math_ops.cast(x, dtypes.int32)