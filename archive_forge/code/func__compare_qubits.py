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
def _compare_qubits(node1, wires1, control1, target1, wires2, control2, target2):
    """Compare the qubit configurations of two operations. The operations are supposed to be similar up to their
    qubits configuration.
    Args:
        node1 (.CommutationDAGNode): First node.
        wires1 (list(int)): Wires of the first node.
        control1 (list(int)): Control wires of the first node.
        target1 (list(int)): Target wires of the first node.
        wires2 (list(int)): Wires of the second node.
        control2 (list(int)): Control wires of the second node.
        target2 (list(int)): Target wires of the second node.
    """
    control_base = {'CNOT': 'PauliX', 'CZ': 'PauliZ', 'CY': 'PauliY', 'CSWAP': 'SWAP', 'Toffoli': 'PauliX', 'ControlledPhaseShift': 'PhaseShift', 'CRX': 'RX', 'CRY': 'RY', 'CRZ': 'RZ', 'CRot': 'Rot', 'MultiControlledX': 'PauliX', 'ControlledOperation': 'ControlledOperation'}
    if control1 and set(control1) == set(control2):
        if control_base[node1.op.name] in symmetric_over_all_wires and set(target1) == set(target2):
            return True
        if target1 == target2:
            return True
    else:
        if node1.op.name in symmetric_over_all_wires and set(wires1) == set(wires2):
            return True
        if wires1 == wires2:
            return True
    return False