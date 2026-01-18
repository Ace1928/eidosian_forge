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
def _initialize_latex_array(self):
    """Initialize qubit and clbit labels and set wire separation"""
    self._img_depth, self._sum_column_widths = self._get_image_depth()
    self._sum_wire_heights = self._img_width
    if self._has_box:
        self._wire_separation = 0.2
    elif self._has_target:
        self._wire_separation = 0.8
    else:
        self._wire_separation = 1.0
    self._latex = [['\\qw' if isinstance(wire, Qubit) else '\\cw' for _ in range(self._img_depth + 1)] for wire in self._wire_map]
    self._latex.append([' '] * (self._img_depth + 1))
    for wire in self._wire_map:
        if isinstance(wire, ClassicalRegister):
            register = wire
            index = self._wire_map[wire]
        else:
            register, bit_index, reg_index = get_bit_reg_index(self._circuit, wire)
            index = bit_index if register is None else reg_index
        wire_label = get_wire_label('latex', register, index, layout=self._layout, cregbundle=self._cregbundle)
        wire_label += ' : '
        if self._initial_state:
            wire_label += '\\ket{{0}}' if isinstance(wire, Qubit) else '0'
        wire_label += ' }'
        if not isinstance(wire, Qubit) and self._cregbundle and (register is not None):
            pos = self._wire_map[register]
            self._latex[pos][1] = '\\lstick{/_{_{' + str(register.size) + '}}} \\cw'
            wire_label = f'\\mathrm{{{wire_label}}}'
        else:
            pos = self._wire_map[wire]
        self._latex[pos][0] = '\\nghost{' + wire_label + ' & ' + '\\lstick{' + wire_label