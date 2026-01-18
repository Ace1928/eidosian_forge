import copy
from typing import Sequence, Callable
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.tape import QuantumTape
from pennylane import transform
@staticmethod
def _circuit(params, gates, initial_circuit):
    """Append parameterized gates to an existing circuit.

        Args:
            params (array[float]): parameters of the gates to be added
            gates (list[Operator]): list of the gates to be added
            initial_circuit (function): user-defined circuit that returns an expectation value

        Returns:
            function: user-defined circuit with appended gates
        """
    final_circuit = append_gate(initial_circuit, params, gates)
    return final_circuit()