import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class ScaledTime(NumberGenerator, TimeDependent):
    """
    The current time multiplied by some conversion factor.
    """
    factor = param.Number(default=1.0, doc='\n       The factor to be multiplied by the current time value.')

    def __call__(self):
        return float(self.time_fn() * self.factor)