from __future__ import annotations
import cmath
import math
import io
import base64
import warnings
from typing import ClassVar, Optional, Type
import logging
import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit, Gate
from qiskit.circuit.library.standard_gates import CXGate, RXGate, RYGate, RZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.two_qubit.weyl import transform_to_magic_basis
from qiskit.synthesis.one_qubit.one_qubit_decompose import (
from qiskit._accelerate import two_qubit_decompose
def _pulse_optimal_chooser(self, best_nbasis, decomposition, target_decomposed) -> QuantumCircuit:
    """Determine method to find pulse optimal circuit. This method may be
        removed once a more general approach is used.

        Returns:
            QuantumCircuit: pulse optimal quantum circuit.
            None: Probably ``nbasis==1`` and original circuit is fine.

        Raises:
            QiskitError: Decomposition for selected basis not implemented.
        """
    circuit = None
    if self.pulse_optimize and best_nbasis in {0, 1}:
        return None
    elif self.pulse_optimize and best_nbasis > 3:
        raise QiskitError(f'Unexpected number of entangling gates ({best_nbasis}) in decomposition.')
    if self._decomposer1q.basis in {'ZSX', 'ZSXX'}:
        if isinstance(self.gate, CXGate):
            if best_nbasis == 3:
                circuit = self._get_sx_vz_3cx_efficient_euler(decomposition, target_decomposed)
            elif best_nbasis == 2:
                circuit = self._get_sx_vz_2cx_efficient_euler(decomposition, target_decomposed)
        else:
            raise QiskitError('pulse_optimizer currently only works with CNOT entangling gate')
    else:
        raise QiskitError(f'"pulse_optimize" currently only works with ZSX basis ({self._decomposer1q.basis} used)')
    return circuit