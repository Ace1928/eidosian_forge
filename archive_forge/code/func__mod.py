from decimal import Decimal
import math
import numbers
import operator
import re
import sys
def _mod(a, b):
    """a % b"""
    da, db = (a.denominator, b.denominator)
    return Fraction(a.numerator * db % (b.numerator * da), da * db)