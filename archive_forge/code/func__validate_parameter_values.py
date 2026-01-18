from __future__ import annotations
from collections.abc import Sequence
import typing
import numpy as np
from qiskit.circuit import QuantumCircuit, ControlFlowOp, Measure
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from ..utils import init_observable
def _validate_parameter_values(parameter_values: Sequence[Sequence[float]] | Sequence[float] | float | None, default: Sequence[Sequence[float]] | Sequence[float] | None=None) -> tuple[tuple[float, ...], ...]:
    if parameter_values is None:
        if default is None:
            raise ValueError('No default `parameter_values`, optional input disallowed.')
        parameter_values = default
    if isinstance(parameter_values, np.ndarray):
        parameter_values = parameter_values.tolist()
    elif isinstance(parameter_values, Sequence):
        parameter_values = tuple((vector.tolist() if isinstance(vector, np.ndarray) else vector for vector in parameter_values))
    if _isreal(parameter_values):
        parameter_values = ((parameter_values,),)
    elif isinstance(parameter_values, Sequence) and (not any((isinstance(vector, Sequence) for vector in parameter_values))):
        parameter_values = (parameter_values,)
    if not isinstance(parameter_values, Sequence) or not all((isinstance(vector, Sequence) for vector in parameter_values)) or (not all((all((_isreal(value) for value in vector)) for vector in parameter_values))):
        raise TypeError('Invalid parameter values, expected Sequence[Sequence[float]].')
    return tuple((tuple((float(value) for value in vector)) for vector in parameter_values))