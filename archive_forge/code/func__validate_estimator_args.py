from __future__ import annotations
from collections.abc import Sequence
import typing
import numpy as np
from qiskit.circuit import QuantumCircuit, ControlFlowOp, Measure
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from ..utils import init_observable
def _validate_estimator_args(circuits: Sequence[QuantumCircuit] | QuantumCircuit, observables: Sequence[BaseOperator | PauliSumOp | str] | BaseOperator | PauliSumOp | str, parameter_values: Sequence[Sequence[float]] | Sequence[float] | float | None=None) -> tuple[tuple[QuantumCircuit], tuple[BaseOperator], tuple[tuple[float]]]:
    """Validate run arguments for a reference Estimator.

    Args:
        circuits: one or more circuit objects.
        observables: one or more observable objects.
        parameter_values: concrete parameters to be bound.

    Returns:
        The formatted arguments ``(circuits, observables, parameter_values)``.

    Raises:
        TypeError: If input arguments are invalid types.
        ValueError: if input arguments are invalid values.
    """
    circuits = _validate_circuits(circuits)
    observables = _validate_observables(observables)
    parameter_values = _validate_parameter_values(parameter_values, default=[()] * len(circuits))
    _cross_validate_circuits_parameter_values(circuits, parameter_values)
    _cross_validate_circuits_observables(circuits, observables)
    return (circuits, observables, parameter_values)