import numpy as np
import pennylane as qml
from pennylane import math
from .single_qubit_unitary import one_qubit_decomposition
def _decomposition_0_cnots(U, wires):
    """If there are no CNOTs, this is just a tensor product of two single-qubit gates.
    We can perform that decomposition directly:
     -╭U- = -A-
     -╰U- = -B-
    """
    A, B = _su2su2_to_tensor_products(U)
    A_ops = one_qubit_decomposition(A, wires[0])
    B_ops = one_qubit_decomposition(B, wires[1])
    return A_ops + B_ops