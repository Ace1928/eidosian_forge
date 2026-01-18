from math import isclose
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.generalized_gates import UCRYGate

        Args:
            num_state_qubits: The number of qubits representing the value to invert.
            scaling: Scaling factor :math:`s` of the reciprocal function, i.e. to compute
                :math:`s / x`.
            neg_vals: Whether :math:`x` might represent negative values. In this case the first
                qubit is the sign, with :math:`|1\rangle` for negative and :math:`|0\rangle` for
                positive.  For the negative case it is assumed that the remaining string represents
                :math:`1 - x`. This is because :math:`e^{-2 \pi i x} = e^{2 \pi i (1 - x)}` for
                :math:`x \in [0,1)`.
            name: The name of the object.

        .. note::

            It is assumed that the binary string :math:`x` represents a number < 1.
        