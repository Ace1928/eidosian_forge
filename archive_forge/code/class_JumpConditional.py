import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class JumpConditional(AbstractInstruction):
    """
    Abstract representation of an conditional jump instruction.
    """
    op: ClassVar[str]

    def __init__(self, target: Union[Label, LabelPlaceholder], condition: MemoryReference):
        if not isinstance(target, (Label, LabelPlaceholder)):
            raise TypeError('target should be a Label')
        if not isinstance(condition, MemoryReference):
            raise TypeError('condition should be an MemoryReference')
        self.target = target
        self.condition = condition

    def out(self) -> str:
        return '%s %s %s' % (self.op, self.target, self.condition)