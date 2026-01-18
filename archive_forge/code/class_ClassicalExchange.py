import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class ClassicalExchange(AbstractInstruction):
    """
    The EXCHANGE instruction.
    """
    op = 'EXCHANGE'

    def __init__(self, left: MemoryReference, right: MemoryReference):
        if not isinstance(left, MemoryReference):
            raise TypeError('left operand should be an MemoryReference')
        if not isinstance(right, MemoryReference):
            raise TypeError('right operand should be an MemoryReference')
        self.left = left
        self.right = right

    def out(self) -> str:
        return '%s %s %s' % (self.op, self.left, self.right)