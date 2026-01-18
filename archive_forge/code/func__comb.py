import math
import warnings
from itertools import combinations_with_replacement
import cupy as cp
def _comb(n, k):
    return math.factorial(n) // (math.factorial(n - k) * math.factorial(k))