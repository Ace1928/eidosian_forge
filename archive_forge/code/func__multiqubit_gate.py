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
def _multiqubit_gate(self, node, node_data, glob_data, xy=None):
    """Draw a gate covering more than one qubit"""
    op = node.op
    if xy is None:
        xy = node_data[node].q_xy
    if isinstance(op, SwapGate):
        self._swap(xy, node, node_data, node_data[node].lc)
        return
    elif isinstance(op, RZZGate):
        self._symmetric_gate(node, node_data, RZZGate, glob_data)
        return
    c_xy = node_data[node].c_xy
    xpos = min((x[0] for x in xy))
    ypos = min((y[1] for y in xy))
    ypos_max = max((y[1] for y in xy))
    if c_xy:
        cxpos = min((x[0] for x in c_xy))
        cypos = min((y[1] for y in c_xy))
        ypos = min(ypos, cypos)
    wid = max(node_data[node].width + 0.21, WID)
    qubit_span = abs(ypos) - abs(ypos_max)
    height = HIG + qubit_span
    box = glob_data['patches_mod'].Rectangle(xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG), width=wid, height=height, fc=node_data[node].fc, ec=node_data[node].ec, linewidth=self._lwidth15, zorder=PORDER_GATE)
    self._ax.add_patch(box)
    for bit, y in enumerate([x[1] for x in xy]):
        self._ax.text(xpos + 0.07 - 0.5 * wid, y, str(bit), ha='left', va='center', fontsize=self._style['fs'], color=node_data[node].gt, clip_on=True, zorder=PORDER_TEXT)
    if c_xy:
        for bit, y in enumerate([x[1] for x in c_xy]):
            self._ax.text(cxpos + 0.07 - 0.5 * wid, y, str(bit), ha='left', va='center', fontsize=self._style['fs'], color=node_data[node].gt, clip_on=True, zorder=PORDER_TEXT)
    if node_data[node].gate_text:
        gate_ypos = ypos + 0.5 * qubit_span
        if node_data[node].param_text:
            gate_ypos = ypos + 0.4 * height
            self._ax.text(xpos + 0.11, ypos + 0.2 * height, node_data[node].param_text, ha='center', va='center', fontsize=self._style['sfs'], color=node_data[node].sc, clip_on=True, zorder=PORDER_TEXT)
        self._ax.text(xpos + 0.11, gate_ypos, node_data[node].gate_text, ha='center', va='center', fontsize=self._style['fs'], color=node_data[node].gt, clip_on=True, zorder=PORDER_TEXT)