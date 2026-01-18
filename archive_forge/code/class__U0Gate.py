import dataclasses
import math
from typing import Iterable, Callable
from qiskit.circuit import (
from qiskit._qasm2 import (  # pylint: disable=no-name-in-module
from .exceptions import QASM2ParseError
class _U0Gate(Gate):

    def __init__(self, count):
        if int(count) != count:
            raise QASM2ParseError('the number of single-qubit delay lengths must be an integer')
        super().__init__('u0', 1, [int(count)])

    def _define(self):
        self._definition = QuantumCircuit(1)
        for _ in [None] * self.params[0]:
            self._definition.id(0)