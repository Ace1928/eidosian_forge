import numpy as np
import rustworkx
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit._accelerate.dense_layout import best_subset
Computes the qubit mapping with the best connectivity.

        Args:
            num_qubits (int): Number of subset qubits to consider.

        Returns:
            ndarray: Array of qubits to use for best connectivity mapping.
        