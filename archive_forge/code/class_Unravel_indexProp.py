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
@register('unravel_index_fallback')
class Unravel_indexProp(operator.CustomOpProp):
    """Fallback unravel_index operator properties."""

    def __init__(self, shape):
        super(Unravel_indexProp, self).__init__(need_top_grad=True)
        self._shape = ast.literal_eval(shape)

    def list_arguments(self):
        return ['indices']

    def infer_shape(self, in_shape):
        dim_list = (1,) if np.isscalar(self._shape) else (len(self._shape),)
        out_shape = dim_list + tuple(in_shape[0])
        return ((in_shape[0],), (out_shape,), ())

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Unravel_index(self._shape)