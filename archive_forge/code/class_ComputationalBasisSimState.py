from typing import List, Sequence, Tuple
import numpy as np
import sympy
import cirq
from cirq.contrib.custom_simulators.custom_state_simulator import CustomStateSimulator
class ComputationalBasisSimState(cirq.SimulationState[ComputationalBasisState]):

    def __init__(self, initial_state, qubits, classical_data):
        state = ComputationalBasisState(cirq.big_endian_int_to_bits(initial_state, bit_count=len(qubits)))
        super().__init__(state=state, qubits=qubits, classical_data=classical_data)

    def _act_on_fallback_(self, action, qubits: Sequence[cirq.Qid], allow_decompose: bool=True):
        gate = action.gate if isinstance(action, cirq.Operation) else action
        if isinstance(gate, cirq.XPowGate):
            i = self.qubit_map[qubits[0]]
            self._state.basis[i] = int(gate.exponent + self._state.basis[i]) % qubits[0].dimension
            return True