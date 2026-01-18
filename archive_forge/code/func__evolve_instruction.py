from __future__ import annotations
import copy
import re
from numbers import Number
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.mixins.tolerances import TolerancesMixin
from qiskit.quantum_info.operators.operator import Operator, BaseOperator
from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit._accelerate.pauli_expval import (
@staticmethod
def _evolve_instruction(statevec, obj, qargs=None):
    """Update the current Statevector by applying an instruction."""
    from qiskit.circuit.reset import Reset
    from qiskit.circuit.barrier import Barrier
    from qiskit.circuit.library.data_preparation.initializer import Initialize
    mat = Operator._instruction_to_matrix(obj)
    if mat is not None:
        return Statevector._evolve_operator(statevec, Operator(mat), qargs=qargs)
    if isinstance(obj, Reset):
        statevec._data = statevec.reset(qargs)._data
        return statevec
    if isinstance(obj, Barrier):
        return statevec
    if isinstance(obj, Initialize):
        if all((isinstance(param, str) for param in obj.params)):
            initialization = Statevector.from_label(''.join(obj.params))._data
        elif len(obj.params) == 1:
            state = int(np.real(obj.params[0]))
            initialization = Statevector.from_int(state, (2,) * obj.num_qubits)._data
        else:
            initialization = np.asarray(obj.params, dtype=complex)
        if qargs is None:
            statevec._data = initialization
        else:
            statevec._data = statevec.reset(qargs)._data
            mat = np.zeros((2 ** len(qargs), 2 ** len(qargs)), dtype=complex)
            mat[:, 0] = initialization
            statevec = Statevector._evolve_operator(statevec, Operator(mat), qargs=qargs)
        return statevec
    if obj.definition is None:
        raise QiskitError(f'Cannot apply Instruction: {obj.name}')
    if not isinstance(obj.definition, QuantumCircuit):
        raise QiskitError('{} instruction definition is {}; expected QuantumCircuit'.format(obj.name, type(obj.definition)))
    if obj.definition.global_phase:
        statevec._data *= np.exp(1j * float(obj.definition.global_phase))
    qubits = {qubit: i for i, qubit in enumerate(obj.definition.qubits)}
    for instruction in obj.definition:
        if instruction.clbits:
            raise QiskitError(f'Cannot apply instruction with classical bits: {instruction.operation.name}')
        if qargs is None:
            new_qargs = [qubits[tup] for tup in instruction.qubits]
        else:
            new_qargs = [qargs[qubits[tup]] for tup in instruction.qubits]
        Statevector._evolve_instruction(statevec, instruction.operation, qargs=new_qargs)
    return statevec