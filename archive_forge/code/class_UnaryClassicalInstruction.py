import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class UnaryClassicalInstruction(AbstractInstruction):
    """
    The abstract class for unary classical instructions.
    """
    op: ClassVar[str]

    def __init__(self, target: MemoryReference):
        if not isinstance(target, MemoryReference):
            raise TypeError('target operand should be an MemoryReference')
        self.target = target

    def out(self) -> str:
        return '%s %s' % (self.op, self.target)