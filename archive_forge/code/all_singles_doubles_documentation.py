import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import BasisState
Returns the expected shape of the tensor that contains the circuit parameters.

        Args:
            singles (Sequence[Sequence]): sequence of lists with the indices of the two qubits
                the :class:`~.pennylane.SingleExcitation` operations act on
            doubles (Sequence[Sequence]): sequence of lists with the indices of the four qubits
                the :class:`~.pennylane.DoubleExcitation` operations act on

        Returns:
            tuple(int): shape of the tensor containing the circuit parameters
        