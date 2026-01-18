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
class DrawElement:
    """An element is an operation that needs to be drawn."""

    def __init__(self, label=None):
        self._width = None
        self.label = self.mid_content = label
        self.top_format = self.mid_format = self.bot_format = '%s'
        self.top_connect = self.bot_connect = ' '
        self.top_pad = self._mid_padding = self.bot_pad = ' '
        self.mid_bck = self.top_bck = self.bot_bck = ' '
        self.bot_connector = {}
        self.top_connector = {}
        self.right_fill = self.left_fill = self.layer_width = 0
        self.wire_label = ''

    @property
    def top(self):
        """Constructs the top line of the element"""
        if self.width % 2 == 0 and len(self.top_format) % 2 == 1 and (len(self.top_connect) == 1):
            ret = self.top_format % (self.top_pad + self.top_connect).center(self.width, self.top_pad)
        else:
            ret = self.top_format % self.top_connect.center(self.width, self.top_pad)
        if self.right_fill:
            ret = ret.ljust(self.right_fill, self.top_pad)
        if self.left_fill:
            ret = ret.rjust(self.left_fill, self.top_pad)
        ret = ret.center(self.layer_width, self.top_bck)
        return ret

    @property
    def mid(self):
        """Constructs the middle line of the element"""
        ret = self.mid_format % self.mid_content.center(self.width, self._mid_padding)
        if self.right_fill:
            ret = ret.ljust(self.right_fill, self._mid_padding)
        if self.left_fill:
            ret = ret.rjust(self.left_fill, self._mid_padding)
        ret = ret.center(self.layer_width, self.mid_bck)
        return ret

    @property
    def bot(self):
        """Constructs the bottom line of the element"""
        if self.width % 2 == 0 and len(self.top_format) % 2 == 1:
            ret = self.bot_format % (self.bot_pad + self.bot_connect).center(self.width, self.bot_pad)
        else:
            ret = self.bot_format % self.bot_connect.center(self.width, self.bot_pad)
        if self.right_fill:
            ret = ret.ljust(self.right_fill, self.bot_pad)
        if self.left_fill:
            ret = ret.rjust(self.left_fill, self.bot_pad)
        ret = ret.center(self.layer_width, self.bot_bck)
        return ret

    @property
    def length(self):
        """Returns the length of the element, including the box around."""
        return max(len(self.top), len(self.mid), len(self.bot))

    @property
    def width(self):
        """Returns the width of the label, including padding"""
        if self._width:
            return self._width
        return len(self.mid_content)

    @width.setter
    def width(self, value):
        self._width = value

    def connect(self, wire_char, where, label=None):
        """Connects boxes and elements using wire_char and setting proper connectors.

        Args:
            wire_char (char): For example '║' or '│'.
            where (list["top", "bot"]): Where the connector should be set.
            label (string): Some connectors have a label (see cu1, for example).
        """
        if 'top' in where and self.top_connector:
            self.top_connect = self.top_connector[wire_char]
        if 'bot' in where and self.bot_connector:
            self.bot_connect = self.bot_connector[wire_char]
        if label:
            self.top_format = self.top_format[:-1] + (label if label else '')