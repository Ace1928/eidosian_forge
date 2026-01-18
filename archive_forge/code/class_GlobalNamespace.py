import collections
import re
import io
import itertools
import numbers
from os.path import dirname, join, abspath
from typing import Iterable, List, Sequence, Union
from qiskit.circuit import (
from qiskit.circuit.bit import Bit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import (
from qiskit.circuit.library import standard_gates
from qiskit.circuit.register import Register
from qiskit.circuit.tools import pi_check
from . import ast
from .experimental import ExperimentalFeatures
from .exceptions import QASM3ExporterError
from .printer import BasicPrinter
class GlobalNamespace:
    """Global namespace dict-like."""
    BASIS_GATE = object()
    qiskit_gates = {'p': standard_gates.PhaseGate, 'x': standard_gates.XGate, 'y': standard_gates.YGate, 'z': standard_gates.ZGate, 'h': standard_gates.HGate, 's': standard_gates.SGate, 'sdg': standard_gates.SdgGate, 't': standard_gates.TGate, 'tdg': standard_gates.TdgGate, 'sx': standard_gates.SXGate, 'rx': standard_gates.RXGate, 'ry': standard_gates.RYGate, 'rz': standard_gates.RZGate, 'cx': standard_gates.CXGate, 'cy': standard_gates.CYGate, 'cz': standard_gates.CZGate, 'cp': standard_gates.CPhaseGate, 'crx': standard_gates.CRXGate, 'cry': standard_gates.CRYGate, 'crz': standard_gates.CRZGate, 'ch': standard_gates.CHGate, 'swap': standard_gates.SwapGate, 'ccx': standard_gates.CCXGate, 'cswap': standard_gates.CSwapGate, 'cu': standard_gates.CUGate, 'CX': standard_gates.CXGate, 'phase': standard_gates.PhaseGate, 'cphase': standard_gates.CPhaseGate, 'id': standard_gates.IGate, 'u1': standard_gates.U1Gate, 'u2': standard_gates.U2Gate, 'u3': standard_gates.U3Gate}
    include_paths = [abspath(join(dirname(__file__), '..', 'qasm', 'libs'))]

    def __init__(self, includelist, basis_gates=()):
        self._data = {gate: self.BASIS_GATE for gate in basis_gates}
        for includefile in includelist:
            if includefile == 'stdgates.inc':
                self._data.update(self.qiskit_gates)
            else:
                pass

    def __setitem__(self, name_str, instruction):
        self._data[name_str] = instruction.base_class
        self._data[id(instruction)] = name_str

    def __getitem__(self, key):
        if isinstance(key, Instruction):
            try:
                return self._data[id(key)]
            except KeyError:
                pass
            if key.name not in self._data:
                raise KeyError(key)
            return key.name
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, instruction):
        if isinstance(instruction, standard_gates.UGate):
            return True
        if id(instruction) in self._data:
            return True
        if self._data.get(instruction.name) is self.BASIS_GATE:
            return True
        if type(instruction) in [Gate, Instruction]:
            return self._data.get(instruction.name, None) == instruction
        type_ = self._data.get(instruction.name)
        if isinstance(type_, type) and isinstance(instruction, type_):
            return True
        return False

    def register(self, instruction):
        """Register an instruction in the namespace"""
        if instruction.name in self._data or (isinstance(instruction, Gate) and (not all((isinstance(param, Parameter) for param in instruction.params)))):
            key = f'{instruction.name}_{id(instruction)}'
        else:
            key = instruction.name
        self[key] = instruction