import math
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_checkpoint_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
Variable initializer.

    Args:
      shape: Shape of `Tensor` to return. Should include OOV on both axes.
      dtype: Must be float32.
      partition_info: variable_scope._PartitionInfo.

    Returns:
      `Tensor` of shape `shape`.

    Raises:
      TypeError: If `dtype` is anything other than float32.
      ValueError: For shape mismatch upon invocation.
    