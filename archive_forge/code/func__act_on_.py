import numpy as np
import pytest
import cirq
def _act_on_(self, sim_state, qubits):
    if isinstance(sim_state, cirq.StabilizerChFormSimulationState):
        qubit = sim_state.qubit_map[qubits[0]]
        sim_state.state.gamma[qubit] += 1
        return True