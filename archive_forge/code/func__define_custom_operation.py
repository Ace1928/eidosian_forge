from __future__ import annotations
import collections.abc
import io
import itertools
import os
import re
import string
from qiskit.circuit import (
from qiskit.circuit.tools import pi_check
from .exceptions import QASM2ExportError
def _define_custom_operation(operation, gates_to_define):
    """Extract a custom definition from the given operation, and append any necessary additional
    subcomponents' definitions to the ``gates_to_define`` ordered dictionary.

    Returns a potentially new :class:`.Instruction`, which should be used for the
    :meth:`~.Instruction.qasm` call (it may have been renamed)."""
    if operation.name in _EXISTING_GATE_NAMES:
        return operation
    escaped = _escape_name(operation.name, 'gate_')
    if escaped != operation.name:
        operation = operation.copy(name=escaped)
    known_good_parameterized = {lib.PhaseGate, lib.RGate, lib.RXGate, lib.RXXGate, lib.RYGate, lib.RYYGate, lib.RZGate, lib.RZXGate, lib.RZZGate, lib.XXMinusYYGate, lib.XXPlusYYGate, lib.UGate, lib.U1Gate, lib.U2Gate, lib.U3Gate}
    if operation.base_class in known_good_parameterized:
        parameterized_operation = type(operation)(*_FIXED_PARAMETERS[:len(operation.params)])
    elif hasattr(operation, '_qasm2_decomposition'):
        new_op = operation._qasm2_decomposition()
        parameterized_operation = operation = new_op.copy(name=_escape_name(new_op.name, 'gate_'))
    else:
        parameterized_operation = operation
    previous_definition_source, _ = gates_to_define.get(operation.name, (None, None))
    if parameterized_operation == previous_definition_source:
        return operation
    if operation.name in gates_to_define:
        operation = _rename_operation(operation)
    new_name = operation.name
    if parameterized_operation.params:
        parameters_qasm = '(' + ','.join((f'param{i}' for i in range(len(parameterized_operation.params)))) + ')'
    else:
        parameters_qasm = ''
    if operation.num_qubits == 0:
        raise QASM2ExportError(f"OpenQASM 2 cannot represent '{operation.name}, which acts on zero qubits.")
    if operation.num_clbits != 0:
        raise QASM2ExportError(f"OpenQASM 2 cannot represent '{operation.name}', which acts on {operation.num_clbits} classical bits.")
    qubits_qasm = ','.join((f'q{i}' for i in range(parameterized_operation.num_qubits)))
    parameterized_definition = getattr(parameterized_operation, 'definition', None)
    if parameterized_definition is None:
        gates_to_define[new_name] = (parameterized_operation, f'opaque {new_name}{parameters_qasm} {qubits_qasm};')
    else:
        qubit_labels = {bit: f'q{i}' for i, bit in enumerate(parameterized_definition.qubits)}
        body_qasm = ' '.join((_custom_operation_statement(instruction, gates_to_define, qubit_labels) for instruction in parameterized_definition.data))
        if operation.name in gates_to_define:
            operation = _rename_operation(operation)
            new_name = operation.name
        definition_qasm = f'gate {new_name}{parameters_qasm} {qubits_qasm} {{ {body_qasm} }}'
        gates_to_define[new_name] = (parameterized_operation, definition_qasm)
    return operation