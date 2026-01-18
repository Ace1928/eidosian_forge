import re
from ... import ndarray, symbol
from .. import HybridBlock, tensor_types
from . import rnn_cell
from ...util import is_np_array
def _register_param(self, name, shape, init, dtype):
    p = self.params.get(name, shape=shape, init=init, allow_deferred_init=True, dtype=dtype)
    setattr(self, name, p)
    return p