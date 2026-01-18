from __future__ import annotations
from typing import Callable, Union
import numbers
import operator
import numpy
import symengine
from qiskit.circuit.exceptions import CircuitError
def _raise_if_passed_nan(self, parameter_values):
    nan_parameter_values = {p: v for p, v in parameter_values.items() if not isinstance(v, numbers.Number)}
    if nan_parameter_values:
        raise CircuitError(f'Expression cannot bind non-numeric values ({nan_parameter_values})')