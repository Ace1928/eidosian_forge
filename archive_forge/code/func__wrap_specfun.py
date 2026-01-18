from ..libmp.backend import xrange
import math
import cmath
@classmethod
def _wrap_specfun(cls, name, f, wrap):
    setattr(cls, name, f)