import math
import numbers
import os
import cupy
from ._util import _get_inttype
def _lcm(a, b):
    return abs(b * (a // math.gcd(a, b)))