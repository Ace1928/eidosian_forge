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
def _instruction_call_site(operation):
    """Return an OpenQASM 2 string for the instruction."""
    if operation.name == 'c3sx':
        qasm2_call = 'c3sqrtx'
    else:
        qasm2_call = operation.name
    if operation.params:
        qasm2_call = '{}({})'.format(qasm2_call, ','.join([pi_check(i, output='qasm', eps=1e-12) for i in operation.params]))
    if operation.condition is not None:
        if not isinstance(operation.condition[0], ClassicalRegister):
            raise QASM2ExportError("OpenQASM 2 can only condition on registers, but got '{operation.condition[0]}'")
        qasm2_call = 'if(%s==%d) ' % (operation.condition[0].name, operation.condition[1]) + qasm2_call
    return qasm2_call