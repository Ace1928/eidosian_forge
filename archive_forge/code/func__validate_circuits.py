from __future__ import annotations
from collections.abc import Sequence
import typing
import numpy as np
from qiskit.circuit import QuantumCircuit, ControlFlowOp, Measure
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from ..utils import init_observable
def _validate_circuits(circuits: Sequence[QuantumCircuit] | QuantumCircuit, requires_measure: bool=False) -> tuple[QuantumCircuit, ...]:
    if isinstance(circuits, QuantumCircuit):
        circuits = (circuits,)
    elif not isinstance(circuits, Sequence) or not all((isinstance(cir, QuantumCircuit) for cir in circuits)):
        raise TypeError('Invalid circuits, expected Sequence[QuantumCircuit].')
    elif not isinstance(circuits, tuple):
        circuits = tuple(circuits)
    if len(circuits) == 0:
        raise ValueError('No circuits were provided.')
    if requires_measure:
        for i, circuit in enumerate(circuits):
            if circuit.num_clbits == 0:
                raise ValueError(f'The {i}-th circuit does not have any classical bit. Sampler requires classical bits, plus measurements on the desired qubits.')
            if not _has_measure(circuit):
                raise ValueError(f'The {i}-th circuit does not have Measure instruction. Without measurements, the circuit cannot be sampled from.')
    return circuits