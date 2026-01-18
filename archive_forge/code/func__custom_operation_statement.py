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
def _custom_operation_statement(instruction, gates_to_define, bit_labels):
    operation = _define_custom_operation(instruction.operation, gates_to_define)
    if instruction.clbits:
        bits = itertools.chain(instruction.qubits, instruction.clbits)
    else:
        bits = instruction.qubits
    bits_qasm = ','.join((bit_labels[j] for j in bits))
    return f'{_instruction_call_site(operation)} {bits_qasm};'