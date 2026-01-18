import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import RZ, RX, CNOT, Hadamard
Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.FermionicSingleExcitation.decomposition`.

        Args:
            weight (float): angle entering the Z rotation
            wires (Any or Iterable[Any]): wires that the operator acts on

        Returns:
            list[.Operator]: decomposition of the operator
        