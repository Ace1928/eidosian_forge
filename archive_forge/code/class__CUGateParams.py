import copy
import math
from cmath import exp
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister
class _CUGateParams(list):
    __slots__ = ('_gate',)

    def __init__(self, gate):
        super().__init__(gate._params)
        self._gate = gate

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._gate._params[key] = value
        if isinstance(key, slice):
            for i, base_key in enumerate(range(*key.indices(4))):
                if base_key < 0:
                    base_key = 4 + base_key
                if base_key < 3:
                    self._gate.base_gate.params[base_key] = value[i]
        else:
            if key < 0:
                key = 4 + key
            if key < 3:
                self._gate.base_gate.params[key] = value