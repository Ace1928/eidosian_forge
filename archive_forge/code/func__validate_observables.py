from __future__ import annotations
from collections.abc import Sequence
import typing
import numpy as np
from qiskit.circuit import QuantumCircuit, ControlFlowOp, Measure
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from ..utils import init_observable
def _validate_observables(observables: Sequence[BaseOperator | PauliSumOp | str] | BaseOperator | PauliSumOp | str) -> tuple[SparsePauliOp, ...]:
    if isinstance(observables, str) or not isinstance(observables, Sequence):
        observables = (observables,)
    if len(observables) == 0:
        raise ValueError('No observables were provided.')
    return tuple((init_observable(obs) for obs in observables))