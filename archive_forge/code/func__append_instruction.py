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
def _append_instruction(self, obj, qargs=None):
    """Update the current Operator by apply an instruction."""
    from qiskit.circuit.barrier import Barrier
    chan = self._instruction_to_superop(obj)
    if chan is not None:
        op = self.compose(chan, qargs=qargs)
        self._data = op.data
    elif isinstance(obj, Barrier):
        return
    else:
        if obj.definition is None:
            raise QiskitError(f'Cannot apply Instruction: {obj.name}')
        if not isinstance(obj.definition, QuantumCircuit):
            raise QiskitError('{} instruction definition is {}; expected QuantumCircuit'.format(obj.name, type(obj.definition)))
        qubit_indices = {bit: idx for idx, bit in enumerate(obj.definition.qubits)}
        for instruction in obj.definition.data:
            if instruction.clbits:
                raise QiskitError(f'Cannot apply instruction with classical bits: {instruction.operation.name}')
            if qargs is None:
                new_qargs = [qubit_indices[tup] for tup in instruction.qubits]
            else:
                new_qargs = [qargs[qubit_indices[tup]] for tup in instruction.qubits]
            self._append_instruction(instruction.operation, qargs=new_qargs)