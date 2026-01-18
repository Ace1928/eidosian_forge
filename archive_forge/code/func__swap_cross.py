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
def _swap_cross(self, xy, color=None):
    """Draw the Swap cross symbol"""
    xpos, ypos = xy
    self._ax.plot([xpos - 0.2 * WID, xpos + 0.2 * WID], [ypos - 0.2 * WID, ypos + 0.2 * WID], color=color, linewidth=self._lwidth2, zorder=PORDER_LINE_PLUS)
    self._ax.plot([xpos - 0.2 * WID, xpos + 0.2 * WID], [ypos + 0.2 * WID, ypos - 0.2 * WID], color=color, linewidth=self._lwidth2, zorder=PORDER_LINE_PLUS)