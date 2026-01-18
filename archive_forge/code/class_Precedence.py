import re
import warnings
from enum import Enum
from math import gcd
class Precedence(Enum):
    """
    Used as Expr.tostring precedence argument.
    """
    ATOM = 0
    POWER = 1
    UNARY = 2
    PRODUCT = 3
    SUM = 4
    LT = 6
    EQ = 7
    LAND = 11
    LOR = 12
    TERNARY = 13
    ASSIGN = 14
    TUPLE = 15
    NONE = 100