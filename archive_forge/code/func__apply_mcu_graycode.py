from math import pi
from typing import Optional, Union, Tuple, List
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library.standard_gates.x import MCXGate
from qiskit.circuit.library.standard_gates.u3 import _generate_gray_code
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.exceptions import QiskitError
def _apply_mcu_graycode(circuit, theta, phi, lam, ctls, tgt, use_basis_gates):
    """Apply multi-controlled u gate from ctls to tgt using graycode
    pattern with single-step angles theta, phi, lam."""
    n = len(ctls)
    gray_code = _generate_gray_code(n)
    last_pattern = None
    for pattern in gray_code:
        if '1' not in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        lm_pos = list(pattern).index('1')
        comp = [i != j for i, j in zip(pattern, last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                circuit.cx(ctls[pos], ctls[lm_pos])
            else:
                indices = [i for i, x in enumerate(pattern) if x == '1']
                for idx in indices[1:]:
                    circuit.cx(ctls[idx], ctls[lm_pos])
        if pattern.count('1') % 2 == 0:
            _apply_cu(circuit, -theta, -lam, -phi, ctls[lm_pos], tgt, use_basis_gates=use_basis_gates)
        else:
            _apply_cu(circuit, theta, phi, lam, ctls[lm_pos], tgt, use_basis_gates=use_basis_gates)
        last_pattern = pattern