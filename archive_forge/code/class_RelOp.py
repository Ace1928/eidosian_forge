import re
import warnings
from enum import Enum
from math import gcd
class RelOp(Enum):
    """
    Used in Op.RELATIONAL expression to specify the function part.
    """
    EQ = 1
    NE = 2
    LT = 3
    LE = 4
    GT = 5
    GE = 6

    @classmethod
    def fromstring(cls, s, language=Language.C):
        if language is Language.Fortran:
            return {'.eq.': RelOp.EQ, '.ne.': RelOp.NE, '.lt.': RelOp.LT, '.le.': RelOp.LE, '.gt.': RelOp.GT, '.ge.': RelOp.GE}[s.lower()]
        return {'==': RelOp.EQ, '!=': RelOp.NE, '<': RelOp.LT, '<=': RelOp.LE, '>': RelOp.GT, '>=': RelOp.GE}[s]

    def tostring(self, language=Language.C):
        if language is Language.Fortran:
            return {RelOp.EQ: '.eq.', RelOp.NE: '.ne.', RelOp.LT: '.lt.', RelOp.LE: '.le.', RelOp.GT: '.gt.', RelOp.GE: '.ge.'}[self]
        return {RelOp.EQ: '==', RelOp.NE: '!=', RelOp.LT: '<', RelOp.LE: '<=', RelOp.GT: '>', RelOp.GE: '>='}[self]