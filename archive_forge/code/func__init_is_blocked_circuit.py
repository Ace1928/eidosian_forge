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
def _init_is_blocked_circuit(self):
    """
        Initialize the list of blocked nodes in the circuit.
        """
    for i in range(0, self.circuit_dag.size):
        self.circuit_blocked[i] = False