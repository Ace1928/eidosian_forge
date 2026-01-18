from typing import List, Sequence, Tuple
import numpy as np
import sympy
import cirq
from cirq.contrib.custom_simulators.custom_state_simulator import CustomStateSimulator
class ComputationalBasisProductState(cirq.qis.QuantumStateRepresentation):

    def __init__(self, initial_state: List[int]):
        self.basis = initial_state

    def copy(self, deep_copy_buffers: bool=True) -> 'ComputationalBasisProductState':
        return ComputationalBasisProductState(self.basis)

    def measure(self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None):
        return [self.basis[i] for i in axes]

    def kron(self, other: 'ComputationalBasisProductState') -> 'ComputationalBasisProductState':
        return ComputationalBasisProductState(self.basis + other.basis)

    def factor(self, axes: Sequence[int], *, validate=True, atol=1e-07) -> Tuple['ComputationalBasisProductState', 'ComputationalBasisProductState']:
        extracted = ComputationalBasisProductState([self.basis[i] for i in axes])
        remainder = ComputationalBasisProductState([self.basis[i] for i in range(len(self.basis)) if i not in axes])
        return (extracted, remainder)

    def reindex(self, axes: Sequence[int]) -> 'ComputationalBasisProductState':
        return ComputationalBasisProductState([self.basis[i] for i in axes])

    @property
    def supports_factor(self) -> bool:
        return True