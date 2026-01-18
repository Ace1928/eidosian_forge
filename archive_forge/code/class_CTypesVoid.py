import ctypes, ctypes.util, operator, sys
from . import model
class CTypesVoid(CTypesData):
    __slots__ = []
    _reftypename = 'void &'

    @staticmethod
    def _from_ctypes(novalue):
        return None

    @staticmethod
    def _to_ctypes(novalue):
        if novalue is not None:
            raise TypeError('None expected, got %s object' % (type(novalue).__name__,))
        return None