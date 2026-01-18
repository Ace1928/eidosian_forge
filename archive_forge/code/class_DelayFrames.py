import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class DelayFrames(AbstractInstruction):

    def __init__(self, frames: List[Frame], duration: float):
        if len(frames) == 0:
            raise ValueError('DELAY expected nonempty list of frames.')
        if len(set((tuple(f.qubits) for f in frames))) != 1:
            raise ValueError('DELAY with explicit frames requires all frames are on the same qubits.')
        self.frames = frames
        self.duration = duration

    def out(self) -> str:
        qubits = self.frames[0].qubits
        ret = 'DELAY ' + _format_qubits_str(qubits)
        for f in self.frames:
            ret += f' "{f.name}"'
        ret += f' {self.duration}'
        return ret