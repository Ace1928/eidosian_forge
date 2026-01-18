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
def _pred_block(self, circuit_sublist, index):
    """It returns the predecessors of a given part of the circuit.
        Args:
            circuit_sublist (list): list of the gates matched in the circuit.
            index (int): Index of the group of matches.
        Returns:
            list: List of predecessors of the current match circuit configuration.
        """
    predecessors = set()
    for node_id in circuit_sublist:
        predecessors = predecessors | set(self.circuit_dag.get_node(node_id).predecessors)
    exclude = set()
    for elem in self.substitution_list[:index]:
        exclude = exclude | set(elem.circuit_config) | set(elem.pred_block)
    pred = list(predecessors - set(circuit_sublist) - exclude)
    pred.sort()
    return pred