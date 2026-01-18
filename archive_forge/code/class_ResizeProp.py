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
@register('resize_fallback')
class ResizeProp(operator.CustomOpProp):
    """Fallback resize operator properties."""

    def __init__(self, new_shape):
        super(ResizeProp, self).__init__(need_top_grad=True)
        self._new_shape = ast.literal_eval(new_shape)

    def list_arguments(self):
        return ['a']

    def infer_shape(self, in_shape):
        out_shape = (self._new_shape,) if np.isscalar(self._new_shape) else self._new_shape
        return ((in_shape[0],), (out_shape,), ())

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Resize(self._new_shape)