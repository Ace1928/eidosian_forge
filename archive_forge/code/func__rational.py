import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
def _rational(self, val):
    """Convert the given value to a rational, if necessary."""
    I32 = 4294967296
    if isinstance(val, int):
        numer, denom = (val, 1)
    elif isinstance(val, fractions.Fraction):
        numer, denom = (val.numerator, val.denominator)
    elif hasattr(val, 'numer'):
        numer, denom = (int(val.numer()), int(val.denom()))
    else:
        param.main.param.log(param.WARNING, "Casting type '%s' to Fraction.fraction" % type(val).__name__)
        frac = fractions.Fraction(str(val))
        numer, denom = (frac.numerator, frac.denominator)
    return (numer % I32, denom % I32)