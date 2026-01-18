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
@register('empty_like_fallback')
class EmptyLikeProp(operator.CustomOpProp):
    """Fallback empty_like operator properties."""

    def __init__(self, dtype, order, subok, shape):
        super(EmptyLikeProp, self).__init__(need_top_grad=True)
        self._dtype = None if dtype == 'None' else dtype
        self._order = order
        self._subok = ast.literal_eval(subok)
        self._shape = ast.literal_eval(shape)

    def list_arguments(self):
        return ['prototype']

    def infer_shape(self, in_shape):
        return ((in_shape[0],), (in_shape[0],), ())

    def infer_type(self, in_type):
        if self._dtype is None:
            return ((in_type[0],), (in_type[0],), ())
        else:
            return ((in_type[0],), (_STR_2_DTYPE_[self._dtype],), ())

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return EmptyLike(self._dtype, self._order, self._subok, self._shape)