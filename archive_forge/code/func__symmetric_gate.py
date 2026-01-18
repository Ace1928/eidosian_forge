import collections
import itertools
import re
from io import StringIO
import numpy as np
from qiskit.circuit import (
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.classical import expr
from qiskit.circuit.annotated_operation import _canonicalize_modifiers, ControlModifier
from qiskit.circuit.library import Initialize
from qiskit.circuit.library.standard_gates import (
from qiskit.qasm3.exporter import QASM3Builder
from qiskit.qasm3.printer import BasicPrinter
from qiskit.circuit.tools.pi_check import pi_check
from qiskit.utils import optionals as _optionals
from .qcstyle import load_style
from ._utils import (
from ..utils import matplotlib_close_if_inline
def _symmetric_gate(self, node, node_data, base_type, glob_data):
    """Draw symmetric gates for cz, cu1, cp, and rzz"""
    op = node.op
    xy = node_data[node].q_xy
    qubit_b = min(xy, key=lambda xy: xy[1])
    qubit_t = max(xy, key=lambda xy: xy[1])
    base_type = getattr(op, 'base_gate', None)
    ec = node_data[node].ec
    tc = node_data[node].tc
    lc = node_data[node].lc
    if not isinstance(op, ZGate) and isinstance(base_type, ZGate):
        num_ctrl_qubits = op.num_ctrl_qubits
        self._ctrl_qubit(xy[-1], glob_data, fc=ec, ec=ec, tc=tc)
        self._line(qubit_b, qubit_t, lc=lc, zorder=PORDER_LINE_PLUS)
    elif isinstance(op, RZZGate) or isinstance(base_type, (U1Gate, PhaseGate, RZZGate)):
        num_ctrl_qubits = 0 if isinstance(op, RZZGate) else op.num_ctrl_qubits
        gate_text = 'P' if isinstance(base_type, PhaseGate) else node_data[node].gate_text
        self._ctrl_qubit(xy[num_ctrl_qubits], glob_data, fc=ec, ec=ec, tc=tc)
        if not isinstance(base_type, (U1Gate, PhaseGate)):
            self._ctrl_qubit(xy[num_ctrl_qubits + 1], glob_data, fc=ec, ec=ec, tc=tc)
        self._sidetext(node, node_data, qubit_b, tc=tc, text=f'{gate_text} ({node_data[node].param_text})')
        self._line(qubit_b, qubit_t, lc=lc)