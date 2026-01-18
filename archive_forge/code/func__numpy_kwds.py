import sys
import types as Types
import warnings
import weakref as Weakref
from inspect import isbuiltin, isclass, iscode, isframe, isfunction, ismethod, ismodule
from math import log
from os import curdir, linesep
from struct import calcsize
from gc import get_objects as _getobjects
from gc import get_referents as _getreferents  # containers only?
from array import array as _array  # array type
def _numpy_kwds(obj):
    t = type(obj)
    if t is _numpy_memmap:
        b, _len_, nb = (144, _len_numpy_memmap, 0)
    else:
        b, _len_, nb = (96, _len_numpy, obj.nbytes)
    return dict(base=_getsizeof(obj, b) - nb, item=_sizeof_Cbyte, leng=_len_, refs=_numpy_refs, vari='itemsize', xtyp=True)