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
def _merge_first_match_and_permutation(list_first_match, permutation):
    """
    Function that returns the final qubits configuration given the first match constraints and the permutation of
    qubits not in the first match.

    Args:
        list_first_match (list): list of qubits indices for the first match.
        permutation (list): possible permutation for the circuit qubits not in the first match.

    Returns:
        list: list of circuit qubits for the given permutation and constraints from the initial match.
    """
    list_circuit = []
    counter = 0
    for elem in list_first_match:
        if elem == -1:
            list_circuit.append(permutation[counter])
            counter = counter + 1
        else:
            list_circuit.append(elem)
    return list_circuit