from decimal import Decimal
import math
import numbers
import operator
import re
import sys
def _floordiv(a, b):
    """a // b"""
    return a.numerator * b.denominator // (a.denominator * b.numerator)