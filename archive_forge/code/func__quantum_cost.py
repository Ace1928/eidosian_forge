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
def _quantum_cost(self, left, right):
    """Compare the two parts of the template and returns True if the quantum cost is reduced.
        Args:
            left (list): list of matched nodes in the template.
            right (list): list of nodes to be replaced.
        Returns:
            bool: True if the quantum cost is reduced
        """
    cost_left = 0
    for i in left:
        cost_left += self.quantum_cost[self.template_dag.get_node(i).op.name]
    cost_right = 0
    for j in right:
        cost_right += self.quantum_cost[self.template_dag.get_node(j).op.name]
    return cost_left > cost_right