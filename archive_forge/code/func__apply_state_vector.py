from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
def _apply_state_vector(self, state, device_wires: Wires):
    """Initialize the internal state vector in a specified state.
            Args:
                state (array[complex]): normalized input state of length ``2**len(wires)``
                    or broadcasted state of shape ``(batch_size, 2**len(wires))``
                device_wires (Wires): wires that get initialized in the state
            """
    if isinstance(state, self._qubit_state.__class__):
        state_data = allocate_aligned_array(state.size, np.dtype(self.C_DTYPE), True)
        state.getState(state_data)
        state = state_data
    ravelled_indices, state = self._preprocess_state_vector(state, device_wires)
    device_wires = self.map_wires(device_wires)
    output_shape = [2] * self.num_wires
    if len(device_wires) == self.num_wires and Wires(sorted(device_wires)) == device_wires:
        state = self._reshape(state, output_shape).ravel(order='C')
        self._qubit_state.UpdateData(state)
        return
    self._qubit_state.setStateVector(ravelled_indices, state)