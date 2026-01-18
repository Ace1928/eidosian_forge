from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class CheckSatResult:
    """Represents the result of a satisfiability check: sat, unsat, unknown.

    >>> s = Solver()
    >>> s.check()
    sat
    >>> r = s.check()
    >>> isinstance(r, CheckSatResult)
    True
    """

    def __init__(self, r):
        self.r = r

    def __deepcopy__(self, memo={}):
        return CheckSatResult(self.r)

    def __eq__(self, other):
        return isinstance(other, CheckSatResult) and self.r == other.r

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        if in_html_mode():
            if self.r == Z3_L_TRUE:
                return '<b>sat</b>'
            elif self.r == Z3_L_FALSE:
                return '<b>unsat</b>'
            else:
                return '<b>unknown</b>'
        elif self.r == Z3_L_TRUE:
            return 'sat'
        elif self.r == Z3_L_FALSE:
            return 'unsat'
        else:
            return 'unknown'

    def _repr_html_(self):
        in_html = in_html_mode()
        set_html_mode(True)
        res = repr(self)
        set_html_mode(in_html)
        return res