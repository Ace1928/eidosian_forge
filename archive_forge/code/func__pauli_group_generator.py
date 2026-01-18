from functools import lru_cache, reduce, singledispatch
from itertools import product
from typing import List, Union
from warnings import warn
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd
from pennylane.tape import OperationRecorder
from pennylane.wires import Wires
def _pauli_group_generator(n_qubits, wire_map=None):
    """Generator function for the Pauli group.

    This function is called by ``pauli_group`` in order to actually generate the
    group elements. They are split so that the outer function can handle input
    validation, while this generator is responsible for performing the actual
    operations.

    Args:
        n_qubits (int): The number of qubits for which to create the group.
        wire_map (dict[Union[str, int], int]): dictionary containing all wire labels
            used in the Pauli word as keys, and unique integer labels as their values.
            If no wire map is provided, wires will be labeled by consecutive integers between :math:`0` and ``n_qubits``.

    Returns:
        .Operation: The next Pauli word in the group.
    """
    element_idx = 0
    if not wire_map:
        wire_map = {wire_idx: wire_idx for wire_idx in range(n_qubits)}
    while element_idx < 4 ** n_qubits:
        binary_string = format(element_idx, f'#0{2 * n_qubits + 2}b')[2:]
        binary_vector = [float(b) for b in binary_string]
        yield binary_to_pauli(binary_vector, wire_map=wire_map)
        element_idx += 1