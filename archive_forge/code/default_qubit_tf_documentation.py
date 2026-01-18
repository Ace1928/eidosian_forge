import itertools
import numpy as np
import semantic_version
import pennylane as qml
from pennylane.math.single_dispatch import _ndim_tf
from . import DefaultQubitLegacy
from .default_qubit_legacy import tolerance
Initialize the internal state vector in a specified state.

        Args:
            state (array[complex]): normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            device_wires (Wires): wires that get initialized in the state

        This implementation only adds a check for parameter broadcasting when initializing
        a quantum state on subsystems of the device.
        