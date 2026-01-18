import warnings
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import SWAP, FermionicSWAP
Returns the shape of the weight tensor required for using parameterized acquaintances in the template.
        Args:
            n_wires (int): Number of qubits
        Returns:
            tuple[int]: shape
        