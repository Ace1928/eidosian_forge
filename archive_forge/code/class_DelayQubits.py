import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class DelayQubits(AbstractInstruction):

    def __init__(self, qubits: List[Union[Qubit, FormalArgument]], duration: float):
        self.qubits = qubits
        self.duration = duration

    def out(self) -> str:
        return f'DELAY {_format_qubits_str(self.qubits)} {self.duration}'