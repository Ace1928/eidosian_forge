from typing import List, Sequence, Tuple
import numpy as np
import sympy
import cirq
from cirq.contrib.custom_simulators.custom_state_simulator import CustomStateSimulator
class ComputationalBasisState(cirq.qis.QuantumStateRepresentation):

    def __init__(self, initial_state: List[int]):
        self.basis = initial_state

    def copy(self, deep_copy_buffers: bool=True) -> 'ComputationalBasisState':
        return ComputationalBasisState(self.basis)

    def measure(self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None):
        return [self.basis[i] for i in axes]