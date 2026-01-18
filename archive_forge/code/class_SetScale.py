import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class SetScale(AbstractInstruction):

    def __init__(self, frame: Frame, scale: ParameterDesignator):
        self.frame = frame
        self.scale = scale

    def out(self) -> str:
        return f'SET-SCALE {self.frame} {self.scale}'

    def get_qubits(self, indices: bool=True) -> Set[QubitDesignator]:
        return _get_frame_qubits(self.frame, indices)