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
class EmptyWire(DrawElement):
    """This element is just the wire, with no operations."""

    def __init__(self, wire):
        super().__init__(wire)
        self._mid_padding = self.mid_bck = wire

    @staticmethod
    def fillup_layer(layer, first_clbit):
        """Given a layer, replace the Nones in it with EmptyWire elements.

        Args:
            layer (list): The layer that contains Nones.
            first_clbit (int): The first wire that is classic.

        Returns:
            list: The new layer, with no Nones.
        """
        for nones in [i for i, x in enumerate(layer) if x is None]:
            layer[nones] = EmptyWire('═') if nones >= first_clbit else EmptyWire('─')
        return layer