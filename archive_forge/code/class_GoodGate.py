from typing import Sequence
import numpy as np
import pytest
import cirq
class GoodGate(cirq.testing.SingleQubitGate):

    def _unitary_(self):
        return np.array([[0, 1], [1, 0]])

    def _act_on_(self, sim_state: 'cirq.SimulationStateBase', qubits: Sequence['cirq.Qid']):
        if isinstance(sim_state, cirq.CliffordTableauSimulationState):
            tableau = sim_state.tableau
            q = sim_state.qubit_map[qubits[0]]
            tableau.rs[:] ^= tableau.zs[:, q]
            return True
        return NotImplemented