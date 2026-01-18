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
class BoxOnWireMid(MultiBox):
    """A generic middle box"""

    def __init__(self, label, input_length, order, wire_label=''):
        super().__init__(label)
        self.top_pad = self.bot_pad = self.top_connect = self.bot_connect = ' '
        self.wire_label = wire_label
        self.left_fill = len(self.wire_label)
        self.top_format = f'│{self.top_pad * self.left_fill} %s │'
        self.bot_format = f'│{self.bot_pad * self.left_fill} %s │'
        self.top_connect = self.bot_connect = self.mid_content = ''
        self.center_label(input_length, order)