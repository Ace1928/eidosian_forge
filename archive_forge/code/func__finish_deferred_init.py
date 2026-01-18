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
def _finish_deferred_init(self):
    """Finishes deferred initialization."""
    if not self._deferred_init:
        return
    init, ctx, default_init, data = self._deferred_init
    self._deferred_init = ()
    assert shape_is_known(self.shape), "Cannot initialize Parameter '%s' because it has invalid shape: %s. Please specify in_units, in_channels, etc for `Block`s." % (self.name, str(self.shape))
    with autograd.pause():
        if data is None:
            kwargs = {'shape': self.shape, 'dtype': self.dtype, 'ctx': context.cpu()}
            if is_np_array():
                if self._stype != 'default':
                    raise ValueError('mxnet.numpy.zeros does not support stype = {}'.format(self._stype))
                zeros_fn = _mx_np.zeros
            else:
                kwargs['stype'] = self._stype
                zeros_fn = ndarray.zeros
            data = zeros_fn(**kwargs)
            initializer.create(default_init)(initializer.InitDesc(self.name, {'__init__': init}), data)
        self._init_impl(data, ctx)