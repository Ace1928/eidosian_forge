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
def _block_shape_tensor(self, spectrum_shape=None):
    if self.block_shape.is_fully_defined():
        return linear_operator_util.shape_tensor(self.block_shape.as_list(), name='block_shape')
    spectrum_shape = array_ops.shape(self.spectrum) if spectrum_shape is None else spectrum_shape
    return spectrum_shape[-self.block_depth:]