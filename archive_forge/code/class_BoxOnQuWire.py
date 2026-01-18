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
class BoxOnQuWire(DrawElement):
    """Draws a box on the quantum wire.

    ::

        top: ┌───┐   ┌───┐
        mid: ┤ A ├ ──┤ A ├──
        bot: └───┘   └───┘
    """

    def __init__(self, label='', top_connect='─', conditional=False):
        super().__init__(label)
        self.top_format = '┌─%s─┐'
        self.mid_format = '┤ %s ├'
        self.bot_format = '└─%s─┘'
        self.top_pad = self.bot_pad = self.mid_bck = '─'
        self.top_connect = top_connect
        self.bot_connect = '╥' if conditional else '─'
        self.mid_content = label
        self.top_connector = {'│': '┴'}
        self.bot_connector = {'│': '┬'}