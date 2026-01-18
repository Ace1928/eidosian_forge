import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops import QubitUnitary
def _make_Z(dim):
    """Calculates the :math:`\\mathcal{Z}` unitary which performs a reflection along the all
    :math:`|0\\rangle` state.

    Args:
        dim (int): dimension of :math:`\\mathcal{Z}`

    Returns:
        array: the :math:`\\mathcal{Z}` unitary
    """
    Z = -np.eye(dim)
    Z[0, 0] = 1
    return Z