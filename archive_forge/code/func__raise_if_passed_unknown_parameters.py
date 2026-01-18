from __future__ import annotations
from typing import Callable, Union
import numbers
import operator
import numpy
import symengine
from qiskit.circuit.exceptions import CircuitError
def _raise_if_passed_unknown_parameters(self, parameters):
    unknown_parameters = parameters - self.parameters
    if unknown_parameters:
        raise CircuitError('Cannot bind Parameters ({}) not present in expression.'.format([str(p) for p in unknown_parameters]))