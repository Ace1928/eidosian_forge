import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class RawInstr(AbstractInstruction):
    """
    A raw instruction represented as a string.
    """

    def __init__(self, instr_str: str):
        if not isinstance(instr_str, str):
            raise TypeError('Raw instructions require a string.')
        self.instr = instr_str

    def out(self) -> str:
        return self.instr

    def __repr__(self) -> str:
        return '<RawInstr {}>'.format(self.instr)