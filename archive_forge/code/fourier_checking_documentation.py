from typing import List
import math
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .generalized_gates.diagonal import Diagonal
Create Fourier checking circuit.

        Args:
            f: truth table for f, length 2**n list of {1,-1}.
            g: truth table for g, length 2**n list of {1,-1}.

        Raises:
            CircuitError: if the inputs f and g are not valid.

        Reference Circuit:
            .. plot::

               from qiskit.circuit.library import FourierChecking
               from qiskit.visualization.library import _generate_circuit_library_visualization
               f = [1, -1, -1, -1]
               g = [1, 1, -1, -1]
               circuit = FourierChecking(f, g)
               _generate_circuit_library_visualization(circuit)
        