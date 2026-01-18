from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
class BinaryExp(Expression):
    operator: ClassVar[str]
    precedence: ClassVar[int]
    associates: ClassVar[str]

    @staticmethod
    def fn(a: ExpressionDesignator, b: ExpressionDesignator) -> Union['BinaryExp', ExpressionValueDesignator]:
        raise NotImplementedError

    def __init__(self, op1: ExpressionDesignator, op2: ExpressionDesignator):
        self.op1 = op1
        self.op2 = op2

    def _substitute(self, d: ParameterSubstitutionsMapDesignator) -> Union['BinaryExp', ExpressionValueDesignator]:
        sop1, sop2 = (substitute(self.op1, d), substitute(self.op2, d))
        return self.fn(sop1, sop2)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.op1 == other.op1 and (self.op2 == other.op2)

    def __neq__(self, other: 'BinaryExp') -> bool:
        return not self.__eq__(other)