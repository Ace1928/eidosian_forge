import re
import numpy as np
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import array_ops_stack as _array_ops_stack
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _fft_size_for_grad(grad, rank):
    return _math_ops.reduce_prod(_array_ops.shape(grad)[-rank:])