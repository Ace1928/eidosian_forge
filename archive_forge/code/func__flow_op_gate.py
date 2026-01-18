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
def _flow_op_gate(self, node, node_data, glob_data):
    """Draw the box for a flow op circuit"""
    xy = node_data[node].q_xy
    xpos = min((x[0] for x in xy))
    ypos = min((y[1] for y in xy))
    ypos_max = max((y[1] for y in xy))
    if_width = node_data[node].width[0] + WID
    box_width = if_width
    for ewidth in node_data[node].width[1:]:
        if ewidth > 0.0:
            box_width += ewidth + WID + 0.3
    qubit_span = abs(ypos) - abs(ypos_max)
    height = HIG + qubit_span
    colors = [self._style['dispcol']['h'][0], self._style['dispcol']['u'][0], self._style['dispcol']['x'][0], self._style['cc']]
    fold_level = 0
    end_x = xpos + box_width
    while end_x > 0.0:
        x_shift = fold_level * self._fold
        y_shift = fold_level * (glob_data['n_lines'] + 1)
        end_x = xpos + box_width - x_shift
        if isinstance(node.op, IfElseOp):
            flow_text = '  If'
        elif isinstance(node.op, WhileLoopOp):
            flow_text = ' While'
        elif isinstance(node.op, ForLoopOp):
            flow_text = ' For'
        elif isinstance(node.op, SwitchCaseOp):
            flow_text = 'Switch'
        if isinstance(node.op, SwitchCaseOp):
            op_spacer = 0.04
            expr_spacer = 0.0
            empty_default_spacer = 0.3 if len(node.op.blocks[-1]) == 0 else 0.0
        else:
            op_spacer = 0.08
            expr_spacer = 0.02
            empty_default_spacer = 0.0
        box = glob_data['patches_mod'].FancyBboxPatch(xy=(xpos - x_shift, ypos - 0.5 * HIG - y_shift), width=box_width + empty_default_spacer, height=height, boxstyle='round, pad=0.1', fc='none', ec=colors[node_data[node].nest_depth % 4], linewidth=self._lwidth3, zorder=PORDER_FLOW)
        self._ax.add_patch(box)
        self._ax.text(xpos - x_shift - op_spacer, ypos_max + 0.2 - y_shift, flow_text, ha='left', va='center', fontsize=self._style['fs'], color=node_data[node].tc, clip_on=True, zorder=PORDER_FLOW)
        self._ax.text(xpos - x_shift + expr_spacer, ypos_max + 0.2 - y_shift - 0.4, node_data[node].expr_text, ha='left', va='center', fontsize=self._style['sfs'], color=node_data[node].tc, clip_on=True, zorder=PORDER_FLOW)
        if isinstance(node.op, ForLoopOp):
            idx_set = str(node_data[node].indexset)
            if 'range' in idx_set:
                idx_set = 'r(' + idx_set[6:-1] + ')'
            else:
                idx_set = str(node_data[node].indexset)[1:-1].split(',')[:5]
                if len(idx_set) > 4:
                    idx_set[4] = '...'
                idx_set = f'{','.join(idx_set)}'
            y_spacer = 0.2 if len(node.qargs) == 1 else 0.5
            self._ax.text(xpos - x_shift - 0.04, ypos_max - y_spacer - y_shift, idx_set, ha='left', va='center', fontsize=self._style['sfs'], color=node_data[node].tc, clip_on=True, zorder=PORDER_FLOW)
        else_case_text = 'Else' if isinstance(node.op, IfElseOp) else 'Case'
        ewidth_incr = if_width
        for circ_num, ewidth in enumerate(node_data[node].width[1:]):
            if ewidth > 0.0:
                self._ax.plot([xpos + ewidth_incr + 0.3 - x_shift, xpos + ewidth_incr + 0.3 - x_shift], [ypos - 0.5 * HIG - 0.08 - y_shift, ypos + height - 0.22 - y_shift], color=colors[node_data[node].nest_depth % 4], linewidth=3.0, linestyle='solid', zorder=PORDER_FLOW)
                self._ax.text(xpos + ewidth_incr + 0.4 - x_shift, ypos_max + 0.2 - y_shift, else_case_text, ha='left', va='center', fontsize=self._style['fs'], color=node_data[node].tc, clip_on=True, zorder=PORDER_FLOW)
                if isinstance(node.op, SwitchCaseOp):
                    jump_val = node_data[node].jump_values[circ_num]
                    if len(str(jump_val)) == 4:
                        jump_text = str(jump_val)[1]
                    elif 'default' in str(jump_val):
                        jump_text = 'default'
                    else:
                        jump_text = str(jump_val)[1:-1].replace(' ', '').split(',')[:5]
                        if len(jump_text) > 4:
                            jump_text[4] = '...'
                        jump_text = f'{', '.join(jump_text)}'
                    y_spacer = 0.2 if len(node.qargs) == 1 else 0.5
                    self._ax.text(xpos + ewidth_incr + 0.4 - x_shift, ypos_max - y_spacer - y_shift, jump_text, ha='left', va='center', fontsize=self._style['sfs'], color=node_data[node].tc, clip_on=True, zorder=PORDER_FLOW)
            ewidth_incr += ewidth + 1
        fold_level += 1