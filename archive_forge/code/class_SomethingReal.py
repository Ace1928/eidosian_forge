import random
from mpmath import *
from mpmath.libmp import *
class SomethingReal:

    def _mpmath_(self, prec, rounding):
        return mp.make_mpf(from_str('1.3', prec, rounding))