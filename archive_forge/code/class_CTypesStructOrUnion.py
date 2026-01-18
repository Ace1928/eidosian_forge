import ctypes, ctypes.util, operator, sys
from . import model
class CTypesStructOrUnion(CTypesBaseStructOrUnion):
    __slots__ = ['_blob']
    _ctype = struct_or_union
    _reftypename = '%s &' % (name,)
    _kind = kind = kind1