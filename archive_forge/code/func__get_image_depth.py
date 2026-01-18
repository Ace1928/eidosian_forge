import io
import itertools
import math
import re
from warnings import warn
import numpy as np
from qiskit.circuit import Clbit, Qubit, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.library.standard_gates import SwapGate, XGate, ZGate, RZZGate, U1Gate, PhaseGate
from qiskit.circuit.measure import Measure
from qiskit.circuit.tools.pi_check import pi_check
from .qcstyle import load_style
from ._utils import (
def _get_image_depth(self):
    """Get depth information for the circuit."""
    columns = 2
    if self._cregbundle and (self._nodes and self._nodes[0] and (self._nodes[0][0].op.name == 'measure' or getattr(self._nodes[0][0].op, 'condition', None))):
        columns += 1
    max_column_widths = []
    for layer in self._nodes:
        column_width = 1
        current_max = 0
        for node in layer:
            op = node.op
            boxed_gates = ['u1', 'u2', 'u3', 'u', 'p', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg', 'rx', 'ry', 'rz', 'ch', 'cy', 'crz', 'cu2', 'cu3', 'cu', 'id']
            target_gates = ['cx', 'ccx', 'cu1', 'cp', 'rzz']
            if op.name in boxed_gates:
                self._has_box = True
            elif op.name in target_gates:
                self._has_target = True
            elif isinstance(op, ControlledGate):
                self._has_box = True
            arg_str_len = 0
            for arg in op.params:
                if not any((isinstance(param, np.ndarray) for param in op.params)):
                    arg_str = re.sub('[-+]?\\d*\\.\\d{2,}|\\d{2,}', self._truncate_float, str(arg))
                    arg_str_len += len(arg_str)
            current_max = max(arg_str_len, current_max)
            base_type = None if not hasattr(op, 'base_gate') else op.base_gate
            if isinstance(op, RZZGate) or isinstance(base_type, (U1Gate, PhaseGate, RZZGate)):
                column_width = 4
        max_column_widths.append(current_max)
        columns += column_width
    sum_column_widths = sum((1 + v / 3 for v in max_column_widths))
    max_wire_name = 3
    for wire in self._wire_map:
        if isinstance(wire, (Qubit, Clbit)):
            register = get_bit_register(self._circuit, wire)
            name = register.name if register is not None else ''
        else:
            name = wire.name
        max_wire_name = max(max_wire_name, len(name))
    sum_column_widths += 5 + max_wire_name / 3
    return (columns, math.ceil(sum_column_widths))