import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class ClassicalLoad(AbstractInstruction):
    """
    The LOAD instruction.
    """
    op = 'LOAD'

    def __init__(self, target: MemoryReference, left: str, right: MemoryReference):
        if not isinstance(target, MemoryReference):
            raise TypeError('target operand should be an MemoryReference')
        if not isinstance(right, MemoryReference):
            raise TypeError('right operand should be an MemoryReference')
        self.target = target
        self.left = left
        self.right = right

    def out(self) -> str:
        return '%s %s %s %s' % (self.op, self.target, self.left, self.right)