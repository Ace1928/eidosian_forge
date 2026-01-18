from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
@dataclass
class WaveformReference(QuilAtom):
    """
    Representation of a Waveform reference.
    """
    name: str
    ' The name of the waveform. '

    def out(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.out()