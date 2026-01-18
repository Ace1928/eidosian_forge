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
def _build_ctrl_gate(self, op, gate_text, wire_list, col):
    """Add a gate with multiple controls to the _latex list"""
    num_cols_op = 1
    num_ctrl_qubits = op.num_ctrl_qubits
    wireqargs = wire_list[num_ctrl_qubits:]
    ctrlqargs = wire_list[:num_ctrl_qubits]
    wire_min = min(wireqargs)
    wire_max = max(wireqargs)
    ctrl_state = f'{op.ctrl_state:b}'.rjust(num_ctrl_qubits, '0')[::-1]
    if len(wireqargs) == 1:
        self._add_controls(wire_list, ctrlqargs, ctrl_state, col)
        if isinstance(op.base_gate, XGate):
            self._latex[wireqargs[0]][col] = '\\targ'
        elif isinstance(op.base_gate, ZGate):
            self._latex[wireqargs[0]][col] = '\\control\\qw'
        elif isinstance(op.base_gate, (U1Gate, PhaseGate)):
            num_cols_op = self._build_symmetric_gate(op, gate_text, wire_list, col)
        else:
            self._latex[wireqargs[0]][col] = '\\gate{%s}' % gate_text
    elif isinstance(op.base_gate, (SwapGate, RZZGate)):
        self._add_controls(wire_list, ctrlqargs, ctrl_state, col)
        num_cols_op = self._build_symmetric_gate(op, gate_text, wire_list, col)
    else:
        for ctrl in ctrlqargs:
            if ctrl in range(wire_min, wire_max):
                wireqargs = wire_list
                break
        else:
            self._add_controls(wire_list, ctrlqargs, ctrl_state, col)
        self._build_multi_gate(op, gate_text, wireqargs, [], col)
    return num_cols_op