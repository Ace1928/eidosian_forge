from typing import Any, cast, Optional, Type
import numpy as np
from cirq.circuits.circuit import Circuit
from cirq.devices import LineQubit
from cirq.ops import common_gates
from cirq.ops.dense_pauli_string import DensePauliString
from cirq import protocols
from cirq.qis import clifford_tableau
from cirq.sim import state_vector_simulation_state, final_state_vector
from cirq.sim.clifford import (
def _final_stabilizer_state_ch_form(circuit: Circuit, qubit_map) -> Optional[stabilizer_state_ch_form.StabilizerStateChForm]:
    """Evolves a default StabilizerStateChForm through the input circuit.

    Initializes a StabilizerStateChForm with default args for the given qubits
    and evolves it by having each operation act on the state.

    Args:
        circuit: An input circuit that acts on the zero state
        qubit_map: A map from qid to the qubit index for the above circuit

    Returns:
        None if any of the operations can not act on a StabilizerStateChForm,
        returns the StabilizerStateChForm otherwise."""
    stabilizer_ch_form = stabilizer_state_ch_form.StabilizerStateChForm(len(qubit_map))
    args = stabilizer_ch_form_simulation_state.StabilizerChFormSimulationState(qubits=list(qubit_map.keys()), prng=np.random.RandomState(), initial_state=stabilizer_ch_form)
    for op in circuit.all_operations():
        try:
            protocols.act_on(op, args, allow_decompose=True)
        except TypeError:
            return None
    return stabilizer_ch_form