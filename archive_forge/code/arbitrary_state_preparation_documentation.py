import functools
import pennylane as qml
from pennylane.operation import Operation, AnyWires
Returns the required shape for the weight tensor.

        Args:
                n_wires (int): number of wires

        Returns:
            tuple[int]: shape
        