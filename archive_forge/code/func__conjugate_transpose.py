from __future__ import annotations
import functools
import itertools
import re
from typing import Literal
import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.library.standard_gates import HGate, IGate, SGate, XGate, YGate, ZGate
from qiskit.circuit.operation import Operation
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, generate_apidocs
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.base_pauli import _count_y
from .base_pauli import BasePauli
from .clifford_circuits import _append_circuit, _append_operation
@staticmethod
def _conjugate_transpose(clifford, method):
    """Return the adjoint, conjugate, or transpose of the Clifford.

        Args:
            clifford (Clifford): a clifford object.
            method (str): what function to apply 'A', 'C', or 'T'.

        Returns:
            Clifford: the modified clifford.
        """
    ret = clifford.copy()
    if method in ['A', 'T']:
        tmp = ret.destab_x.copy()
        ret.destab_x = ret.stab_z.T
        ret.destab_z = ret.destab_z.T
        ret.stab_x = ret.stab_x.T
        ret.stab_z = tmp.T
        ret.phase ^= clifford.dot(ret).phase
    if method in ['C', 'T']:
        ret.phase ^= np.mod(_count_y(ret.x, ret.z), 2).astype(bool)
    return ret