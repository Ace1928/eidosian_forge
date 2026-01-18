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
def _draw_regs_wires(self, num_folds, xmax, max_x_index, qubits_dict, clbits_dict, glob_data):
    """Draw the register names and numbers, wires, and vertical lines at the ends"""
    for fold_num in range(num_folds + 1):
        for qubit in qubits_dict.values():
            qubit_label = qubit['wire_label']
            y = qubit['y'] - fold_num * (glob_data['n_lines'] + 1)
            self._ax.text(glob_data['x_offset'] - 0.2, y, qubit_label, ha='right', va='center', fontsize=1.25 * self._style['fs'], color=self._style['tc'], clip_on=True, zorder=PORDER_TEXT)
            self._line([glob_data['x_offset'], y], [xmax, y], zorder=PORDER_REGLINE)
        this_clbit_dict = {}
        for clbit in clbits_dict.values():
            y = clbit['y'] - fold_num * (glob_data['n_lines'] + 1)
            if y not in this_clbit_dict.keys():
                this_clbit_dict[y] = {'val': 1, 'wire_label': clbit['wire_label'], 'register': clbit['register']}
            else:
                this_clbit_dict[y]['val'] += 1
        for y, this_clbit in this_clbit_dict.items():
            if self._cregbundle and this_clbit['register'] is not None:
                self._ax.plot([glob_data['x_offset'] + 0.2, glob_data['x_offset'] + 0.3], [y - 0.1, y + 0.1], color=self._style['cc'], zorder=PORDER_REGLINE)
                self._ax.text(glob_data['x_offset'] + 0.1, y + 0.1, str(this_clbit['register'].size), ha='left', va='bottom', fontsize=0.8 * self._style['fs'], color=self._style['tc'], clip_on=True, zorder=PORDER_TEXT)
            self._ax.text(glob_data['x_offset'] - 0.2, y, this_clbit['wire_label'], ha='right', va='center', fontsize=1.25 * self._style['fs'], color=self._style['tc'], clip_on=True, zorder=PORDER_TEXT)
            self._line([glob_data['x_offset'], y], [xmax, y], lc=self._style['cc'], ls=self._style['cline'], zorder=PORDER_REGLINE)
        feedline_r = num_folds > 0 and num_folds > fold_num
        feedline_l = fold_num > 0
        if feedline_l or feedline_r:
            xpos_l = glob_data['x_offset'] - 0.01
            xpos_r = self._fold + glob_data['x_offset'] + 0.1
            ypos1 = -fold_num * (glob_data['n_lines'] + 1)
            ypos2 = -(fold_num + 1) * glob_data['n_lines'] - fold_num + 1
            if feedline_l:
                self._ax.plot([xpos_l, xpos_l], [ypos1, ypos2], color=self._style['lc'], linewidth=self._lwidth15, zorder=PORDER_REGLINE)
            if feedline_r:
                self._ax.plot([xpos_r, xpos_r], [ypos1, ypos2], color=self._style['lc'], linewidth=self._lwidth15, zorder=PORDER_REGLINE)
        box = glob_data['patches_mod'].Rectangle(xy=(glob_data['x_offset'] - 0.1, -fold_num * (glob_data['n_lines'] + 1) + 0.5), width=-25.0, height=-(fold_num + 1) * (glob_data['n_lines'] + 1), fc=self._style['bg'], ec=self._style['bg'], linewidth=self._lwidth15, zorder=PORDER_MASK)
        self._ax.add_patch(box)
    if self._style['index']:
        for layer_num in range(max_x_index):
            if self._fold > 0:
                x_coord = layer_num % self._fold + glob_data['x_offset'] + 0.53
                y_coord = -(layer_num // self._fold) * (glob_data['n_lines'] + 1) + 0.65
            else:
                x_coord = layer_num + glob_data['x_offset'] + 0.53
                y_coord = 0.65
            self._ax.text(x_coord, y_coord, str(layer_num + 1), ha='center', va='center', fontsize=self._style['sfs'], color=self._style['tc'], clip_on=True, zorder=PORDER_TEXT)