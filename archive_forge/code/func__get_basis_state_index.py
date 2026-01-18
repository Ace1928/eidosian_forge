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
def _get_basis_state_index(self, state, wires):
    """Returns the basis state index of a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s
            wires (Wires): wires that the provided computational state should be initialized on

        Returns:
            int: basis state index
        """
    device_wires = self.map_wires(wires)
    n_basis_state = len(state)
    if not set(state.tolist()).issubset({0, 1}):
        raise ValueError('BasisState parameter must consist of 0 or 1 integers.')
    if n_basis_state != len(device_wires):
        raise ValueError('BasisState parameter and wires must be of equal length.')
    basis_states = 2 ** (self.num_wires - 1 - np.array(device_wires))
    basis_states = qml.math.convert_like(basis_states, state)
    return int(qml.math.dot(state, basis_states))