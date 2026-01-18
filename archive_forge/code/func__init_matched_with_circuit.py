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
def _init_matched_with_circuit(self):
    """
        Initialize the list of corresponding matches in the pattern for the circuit.
        """
    for i in range(0, self.circuit_dag.size):
        if i == self.node_id_c:
            self.circuit_matched_with[i] = [self.node_id_p]
        else:
            self.circuit_matched_with[i] = []