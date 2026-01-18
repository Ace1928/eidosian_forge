import math
import numbers
import random
import sys
from fractions import Fraction
from decimal import Decimal
from itertools import groupby, repeat
from bisect import bisect_left, bisect_right
from math import hypot, sqrt, fabs, exp, erf, tau, log, fsum
from functools import reduce
from operator import mul
from collections import Counter, namedtuple, defaultdict
def _integer_sqrt_of_frac_rto(n: int, m: int) -> int:
    """Square root of n/m, rounded to the nearest integer using round-to-odd."""
    a = math.isqrt(n // m)
    return a | (a * a * m != n)