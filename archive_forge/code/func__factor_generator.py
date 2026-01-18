from __future__ import annotations
import operator
from functools import reduce
from math import sqrt
import numpy as np
from scipy.special import erf
def _factor_generator(n: int) -> dict[int, int]:
    """
    From a given natural integer, returns the prime factors and their multiplicity

    Args:
        n: Natural integer
    """
    p = prime_factors(n)
    factors: dict[int, int] = {}
    for p1 in p:
        try:
            factors[p1] += 1
        except KeyError:
            factors[p1] = 1
    return factors