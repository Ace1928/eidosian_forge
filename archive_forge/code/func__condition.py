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
def _condition(self, node, node_data, wire_map, outer_circuit, cond_xy, glob_data):
    """Add a conditional to a gate"""
    if isinstance(node.op, SwitchCaseOp):
        if isinstance(node.op.target, expr.Expr):
            condition = node.op.target
        elif isinstance(node.op.target, Clbit):
            condition = (node.op.target, 1)
        else:
            condition = (node.op.target, 2 ** node.op.target.size - 1)
    else:
        condition = node.op.condition
    override_fc = False
    first_clbit = len(self._qubits)
    cond_pos = []
    if isinstance(condition, expr.Expr):
        condition_bits = condition_resources(condition).clbits
        label = '[expr]'
        override_fc = True
        registers = collections.defaultdict(list)
        for bit in condition_bits:
            registers[get_bit_register(outer_circuit, bit)].append(bit)
        cond_pos.extend((cond_xy[wire_map[bit] - first_clbit] for bit in registers.pop(None, ())))
        if self._cregbundle:
            cond_pos.extend((cond_xy[wire_map[register] - first_clbit] for register in registers))
        else:
            cond_pos.extend((cond_xy[wire_map[bit] - first_clbit] for bit in itertools.chain.from_iterable(registers.values())))
        val_bits = ['1'] * len(cond_pos)
    else:
        label, val_bits = get_condition_label_val(condition, self._circuit, self._cregbundle)
        cond_bit_reg = condition[0]
        cond_bit_val = int(condition[1])
        override_fc = cond_bit_val != 0 and self._cregbundle and isinstance(cond_bit_reg, ClassicalRegister)
        if not self._cregbundle and isinstance(cond_bit_reg, ClassicalRegister):
            for idx in range(cond_bit_reg.size):
                cond_pos.append(cond_xy[wire_map[cond_bit_reg[idx]] - first_clbit])
        elif self._cregbundle and isinstance(cond_bit_reg, Clbit):
            register = get_bit_register(outer_circuit, cond_bit_reg)
            if register is not None:
                cond_pos.append(cond_xy[wire_map[register] - first_clbit])
            else:
                cond_pos.append(cond_xy[wire_map[cond_bit_reg] - first_clbit])
        else:
            cond_pos.append(cond_xy[wire_map[cond_bit_reg] - first_clbit])
    xy_plot = []
    for val_bit, xy in zip(val_bits, cond_pos):
        fc = self._style['lc'] if override_fc or val_bit == '1' else self._style['bg']
        box = glob_data['patches_mod'].Circle(xy=xy, radius=WID * 0.15, fc=fc, ec=self._style['lc'], linewidth=self._lwidth15, zorder=PORDER_GATE)
        self._ax.add_patch(box)
        xy_plot.append(xy)
    qubit_b = min(node_data[node].q_xy, key=lambda xy: xy[1])
    clbit_b = min(xy_plot, key=lambda xy: xy[1])
    if isinstance(node.op, (IfElseOp, WhileLoopOp, SwitchCaseOp)):
        qubit_b = (qubit_b[0], qubit_b[1] - (0.5 * HIG + 0.14))
    xpos, ypos = clbit_b
    if isinstance(node.op, Measure):
        xpos += 0.3
    self._ax.text(xpos, ypos - 0.3 * HIG, label, ha='center', va='top', fontsize=self._style['sfs'], color=self._style['tc'], clip_on=True, zorder=PORDER_TEXT)
    self._line(qubit_b, clbit_b, lc=self._style['cc'], ls=self._style['cline'])