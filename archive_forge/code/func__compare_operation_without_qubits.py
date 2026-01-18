import copy
import itertools
from collections import OrderedDict
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
from pennylane import adjoint
from pennylane.ops.qubit.attributes import symmetric_over_all_wires
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.wires import Wires
def _compare_operation_without_qubits(node_1, node_2):
    """Compare two operations without taking the qubits into account.

    Args:
        node_1 (.CommutationDAGNode): First operation.
        node_2 (.CommutationDAGNode): Second operation.
    Return:
        Bool: True if similar operation (no qubits comparison) and False otherwise.
    """
    return node_1.op.name == node_2.op.name and node_1.op.data == node_2.op.data