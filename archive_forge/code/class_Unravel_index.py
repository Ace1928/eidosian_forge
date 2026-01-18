from distutils.version import StrictVersion
import functools
import ast
import numpy as np
from . import operator
from . import numpy as _mx_np  # pylint: disable=reimported
from .util import np_array, use_np
from .numpy.utils import _STR_2_DTYPE_
from .ndarray.numpy import _internal as _nd_npi
from .symbol.numpy import _internal as _sym_npi
@use_np
class Unravel_index(operator.CustomOp):
    """Fallback to NumPy Unravel_index operator."""

    def __init__(self, shape):
        super(Unravel_index, self).__init__()
        self._shape = shape

    def forward(self, is_train, req, in_data, out_data, aux):
        out = np.unravel_index(in_data[0].asnumpy(), self._shape)
        self.assign(out_data[0], req[0], _mx_np.array(out, dtype=out[0].dtype, ctx=out_data[0].ctx))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        raise NotImplementedError('Operator Unravel_index does not support gradient computation')