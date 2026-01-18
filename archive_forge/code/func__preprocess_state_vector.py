from itertools import islice, product
from typing import List
import numpy as np
import pennylane as qml
from pennylane import BasisState, QubitDevice, StatePrep
from pennylane.devices import DefaultQubitLegacy
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.wires import Wires
from ._serialize import QuantumScriptSerializer
from ._version import __version__
def _preprocess_state_vector(self, state, device_wires):
    """Initialize the internal state vector in a specified state.

        Args:
            state (array[complex]): normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            device_wires (Wires): wires that get initialized in the state

        Returns:
            array[complex]: normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            array[int]: indices for which the state is changed to input state vector elements
        """
    device_wires = self.map_wires(device_wires)
    if state.dtype.kind == 'i':
        state = qml.numpy.array(state, dtype=self.C_DTYPE)
    state = self._asarray(state, dtype=self.C_DTYPE)
    if len(device_wires) == self.num_wires and Wires(sorted(device_wires)) == device_wires:
        return (None, state)
    basis_states = np.array(list(product([0, 1], repeat=len(device_wires))))
    unravelled_indices = np.zeros((2 ** len(device_wires), self.num_wires), dtype=int)
    unravelled_indices[:, device_wires] = basis_states
    ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)
    return (ravelled_indices, state)