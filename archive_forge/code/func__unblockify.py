import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util.tf_export import tf_export
def _unblockify(self, x):
    """Flatten the trailing block dimensions."""
    if x.shape.is_fully_defined():
        x_shape = x.shape.as_list()
        x_leading_shape = x_shape[:-self.block_depth]
        x_block_shape = x_shape[-self.block_depth:]
        flat_shape = x_leading_shape + [np.prod(x_block_shape)]
    else:
        x_shape = array_ops.shape(x)
        x_leading_shape = x_shape[:-self.block_depth]
        x_block_shape = x_shape[-self.block_depth:]
        flat_shape = array_ops.concat((x_leading_shape, [math_ops.reduce_prod(x_block_shape)]), 0)
    return array_ops.reshape(x, flat_shape)