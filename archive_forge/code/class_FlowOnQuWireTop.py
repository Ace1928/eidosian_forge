from io import StringIO
from warnings import warn
from shutil import get_terminal_size
import collections
import sys
from qiskit.circuit import Qubit, Clbit, ClassicalRegister
from qiskit.circuit import ControlledGate, Reset, Measure
from qiskit.circuit import ControlFlowOp, WhileLoopOp, IfElseOp, ForLoopOp, SwitchCaseOp
from qiskit.circuit.classical import expr
from qiskit.circuit.controlflow import node_resources
from qiskit.circuit.library.standard_gates import IGate, RZZGate, SwapGate, SXGate, SXdgGate
from qiskit.circuit.annotated_operation import _canonicalize_modifiers, ControlModifier
from qiskit.circuit.tools.pi_check import pi_check
from qiskit.qasm3.exporter import QASM3Builder
from qiskit.qasm3.printer import BasicPrinter
from ._utils import (
from ..exceptions import VisualizationError
class FlowOnQuWireTop(MultiBox, BoxOnQuWire):
    """Draws the top of a box for a ControlFlowOp that uses more than one qubit."""

    def __init__(self, section, label='', top_connect=None, wire_label=''):
        super().__init__(label)
        self.wire_label = wire_label
        self.bot_connect = self.bot_pad = ' '
        self.mid_content = ''
        self.left_fill = len(self.wire_label)
        if section == CF_RIGHT:
            self.top_format = 's'.center(self.left_fill + 2, '─') + '─┐'
            self.top_format = self.top_format.replace('s', '%s')
            self.mid_format = f' {self.wire_label} %s ├'
            self.bot_format = f' {self.bot_pad * self.left_fill} %s │'
        else:
            self.top_format = '┌─' + 's'.center(self.left_fill + 2, '─') + ' '
            self.top_format = self.top_format.replace('s', '%s')
            self.mid_format = f'┤{self.wire_label} %s  '
            self.bot_format = f'│{self.bot_pad * self.left_fill} %s  '
        self.top_connect = top_connect if top_connect else '─'