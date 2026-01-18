import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class ArithmeticBinaryOp(AbstractInstruction):
    """
    The abstract class for binary arithmetic classical instructions.
    """
    op: ClassVar[str]

    def __init__(self, left: MemoryReference, right: Union[MemoryReference, int, float]):
        if not isinstance(left, MemoryReference):
            raise TypeError('left operand should be an MemoryReference')
        if not isinstance(right, MemoryReference) and (not isinstance(right, int)) and (not isinstance(right, float)):
            raise TypeError('right operand should be an MemoryReference or a numeric literal')
        self.left = left
        self.right = right

    def out(self) -> str:
        return '%s %s %s' % (self.op, self.left, self.right)