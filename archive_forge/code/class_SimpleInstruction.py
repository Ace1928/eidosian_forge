import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class SimpleInstruction(AbstractInstruction):
    """
    Abstract class for simple instructions with no arguments.
    """
    op: ClassVar[str]

    def out(self) -> str:
        return self.op