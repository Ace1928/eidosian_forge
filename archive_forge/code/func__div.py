from decimal import Decimal
import math
import numbers
import operator
import re
import sys
def _div(a, b):
    """a / b"""
    na, da = (a.numerator, a.denominator)
    nb, db = (b.numerator, b.denominator)
    g1 = math.gcd(na, nb)
    if g1 > 1:
        na //= g1
        nb //= g1
    g2 = math.gcd(db, da)
    if g2 > 1:
        da //= g2
        db //= g2
    n, d = (na * db, nb * da)
    if d < 0:
        n, d = (-n, -d)
    return Fraction(n, d, _normalize=False)