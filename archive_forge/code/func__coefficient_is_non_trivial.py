import re
import operator
from fractions import Fraction
import sys
def _coefficient_is_non_trivial(c):
    if isinstance(c, Polynomial):
        return c._monomials
    return not c == 0