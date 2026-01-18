import ctypes, ctypes.util, operator, sys
from . import model
class CTypesGenericArray(CTypesData):
    __slots__ = []

    @classmethod
    def _newp(cls, init):
        return cls(init)

    def __iter__(self):
        for i in xrange(len(self)):
            yield self[i]

    def _get_own_repr(self):
        return self._addr_repr(ctypes.addressof(self._blob))