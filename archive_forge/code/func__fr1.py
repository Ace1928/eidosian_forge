import warnings
from .._utils import set_module
from ._machar import MachAr
from . import numeric
from . import numerictypes as ntypes
from .numeric import array, inf, NaN
from .umath import log10, exp2, nextafter, isnan
def _fr1(a):
    """fix rank > 0 --> rank-0"""
    if a.size == 1:
        a = a.copy()
        a.shape = ()
    return a