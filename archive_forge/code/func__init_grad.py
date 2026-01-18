from collections import OrderedDict, defaultdict
import warnings
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, context
from ..context import Context, cpu
from .. import autograd
from .utils import _indent, _brief_print_list, shape_is_known
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def _init_grad(self):
    """Initialize grad buffers."""
    if self.grad_req == 'null':
        self._grad = None
        return
    if is_np_array():
        if self._grad_stype != 'default':
            raise ValueError('mxnet.numpy.zeros does not support stype = {}'.format(self._grad_stype))
        self._grad = [_mx_np.zeros(shape=i.shape, dtype=i.dtype, ctx=i.ctx) for i in self._data]
    else:
        self._grad = [ndarray.zeros(shape=i.shape, dtype=i.dtype, ctx=i.ctx, stype=self._grad_stype) for i in self._data]
    autograd.mark_variables(self._check_and_get(self._data, list), self._grad, self.grad_req)