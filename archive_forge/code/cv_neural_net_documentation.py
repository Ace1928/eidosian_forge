import pennylane as qml
from pennylane.operation import Operation, AnyWires
Returns a list of shapes for the 11 parameter tensors.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of wires

        Returns:
            list[tuple[int]]: list of shapes
        