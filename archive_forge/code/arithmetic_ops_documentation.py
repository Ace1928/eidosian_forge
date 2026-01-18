from copy import copy
import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.wires import Wires
from pennylane.ops import Identity
Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.IntegerComparator.decomposition`.

        Args:
            value (int): The value :math:`L` that the state's decimal representation is compared against.
            geq (bool): If set to ``True``, the comparison made will be :math:`n \geq L`. If ``False``, the comparison
                made will be :math:`n < L`.
            wires (Union[Wires, Sequence[int], or int]): Control wire(s) followed by a single target wire where
                the operation acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.IntegerComparator.compute_decomposition(4, wires=[0, 1, 2, 3]))
        [MultiControlledX(wires=[0, 1, 2, 3], control_values="100"),
         MultiControlledX(wires=[0, 1, 2, 3], control_values="101"),
         MultiControlledX(wires=[0, 1, 2, 3], control_values="110"),
         MultiControlledX(wires=[0, 1, 2, 3], control_values="111")]
        