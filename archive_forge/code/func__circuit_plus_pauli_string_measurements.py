import collections
from typing import cast, Dict, Optional, Union, TYPE_CHECKING
import numpy as np
from cirq import ops
from cirq.work import collector
def _circuit_plus_pauli_string_measurements(circuit: 'cirq.AbstractCircuit', pauli_string: 'cirq.PauliString') -> 'cirq.AbstractCircuit':
    """A circuit measuring the given observable at the end of the given circuit."""
    assert pauli_string
    return circuit.from_moments(*circuit, pauli_string.to_z_basis_ops(), ops.measure(*sorted(pauli_string.keys()), key='out'))