import operator
from math import gcd
from decimal import Decimal
from fractions import Fraction
import sys
from typing import Tuple as tTuple, Type
@classmethod
def _new_check(cls, numerator, denominator):
    """Construct PythonMPQ, check divide by zero and canonicalize signs"""
    if not denominator:
        raise ZeroDivisionError(f'Zero divisor {numerator}/{denominator}')
    elif denominator < 0:
        numerator = -numerator
        denominator = -denominator
    return cls._new(numerator, denominator)