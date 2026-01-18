import sys
from ctypes import *
class BigEndianUnion(Union, metaclass=_swapped_union_meta):
    """Union with big endian byte order"""
    __slots__ = ()
    _swappedbytes_ = None