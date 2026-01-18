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
Evolves a default StabilizerStateChForm through the input circuit.

    Initializes a StabilizerStateChForm with default args for the given qubits
    and evolves it by having each operation act on the state.

    Args:
        circuit: An input circuit that acts on the zero state
        qubit_map: A map from qid to the qubit index for the above circuit

    Returns:
        None if any of the operations can not act on a StabilizerStateChForm,
        returns the StabilizerStateChForm otherwise.