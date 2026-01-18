import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
class DictIter(object):
    """A iterator for the `Dict.items()`.

    Only the `.items()` is needed.  `.keys` and `.values` can be trivially
    implemented on the `.items` iterator.
    """

    def __init__(self, parent):
        self.parent = parent
        itsize = self.parent.tc.numba_dict_iter_sizeof()
        self.it_state_buf = (ctypes.c_char_p * itsize)(0)
        self.it = ctypes.cast(self.it_state_buf, ctypes.c_void_p)
        self.parent.dict_iter(self.it)

    def __iter__(self):
        return self

    def __next__(self):
        out = self.parent.dict_iter_next(self.it)
        if out is None:
            raise StopIteration
        else:
            k, v = out
            return (k.decode(), v.decode())
    next = __next__