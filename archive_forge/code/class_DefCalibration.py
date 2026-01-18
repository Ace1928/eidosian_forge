import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class DefCalibration(AbstractInstruction):

    def __init__(self, name: str, parameters: List[ParameterDesignator], qubits: List[Union[Qubit, FormalArgument]], instrs: List[AbstractInstruction]):
        self.name = name
        self.parameters = parameters
        self.qubits = qubits
        self.instrs = instrs

    def out(self) -> str:
        ret = f'DEFCAL {self.name}'
        if len(self.parameters) > 0:
            ret += _format_params(self.parameters)
        ret += ' ' + _format_qubits_str(self.qubits) + ':\n'
        for instr in self.instrs:
            ret += f'    {instr.out()}\n'
        return ret