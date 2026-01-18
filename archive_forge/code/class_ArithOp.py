import re
import warnings
from enum import Enum
from math import gcd
class ArithOp(Enum):
    """
    Used in Op.APPLY expression to specify the function part.
    """
    POS = 1
    NEG = 2
    ADD = 3
    SUB = 4
    MUL = 5
    DIV = 6
    POW = 7