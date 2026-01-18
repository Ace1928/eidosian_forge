import random
from mpmath import *
from mpmath.libmp import *
class mympf:

    @property
    def _mpf_(self):
        return mpf(3.5)._mpf_