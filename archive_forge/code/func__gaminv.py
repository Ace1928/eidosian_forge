import math
import numpy as np
from scipy import special
from scipy.stats._qmc import primes_from_2_to
def _gaminv(a, b):
    return special.gammaincinv(b, a)