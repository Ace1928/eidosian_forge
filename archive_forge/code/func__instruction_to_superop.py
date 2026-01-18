from __future__ import annotations
import copy
from typing import TYPE_CHECKING
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.transformations import _bipartite_tensor, _to_superop
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
@classmethod
def _instruction_to_superop(cls, obj):
    """Return superop for instruction if defined or None otherwise."""
    if not isinstance(obj, Instruction):
        raise QiskitError('Input is not an instruction.')
    chan = None
    if obj.name == 'reset':
        chan = SuperOp(np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    if obj.name == 'kraus':
        kraus = obj.params
        dim = len(kraus[0])
        chan = SuperOp(_to_superop('Kraus', (kraus, None), dim, dim))
    elif hasattr(obj, 'to_matrix'):
        try:
            kraus = [obj.to_matrix()]
            dim = len(kraus[0])
            chan = SuperOp(_to_superop('Kraus', (kraus, None), dim, dim))
        except QiskitError:
            pass
    return chan