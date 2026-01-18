import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.qchem.givens_decomposition import givens_decomposition
Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.BasisRotation.decomposition`.

        Args:
            wires (Any or Iterable[Any]): wires that the operator acts on
            unitary_matrix (array): matrix specifying the basis transformation
            check (bool): test unitarity of the provided `unitary_matrix`

        Returns:
            list[.Operator]: decomposition of the operator
        