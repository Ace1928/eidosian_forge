import re
import warnings
from enum import Enum
from math import gcd
class _Pair:

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def substitute(self, symbols_map):
        left, right = (self.left, self.right)
        if isinstance(left, Expr):
            left = left.substitute(symbols_map)
        if isinstance(right, Expr):
            right = right.substitute(symbols_map)
        return _Pair(left, right)

    def __repr__(self):
        return f'{type(self).__name__}({self.left}, {self.right})'