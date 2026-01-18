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
def _set_ctrl_bits(self, ctrl_state, num_ctrl_qubits, qbit, glob_data, ec=None, tc=None, text='', qargs=None):
    """Determine which qubits are controls and whether they are open or closed"""
    if text:
        qlist = [self._circuit.find_bit(qubit).index for qubit in qargs]
        ctbits = qlist[:num_ctrl_qubits]
        qubits = qlist[num_ctrl_qubits:]
        max_ctbit = max(ctbits)
        min_ctbit = min(ctbits)
        top = min(qubits) > min_ctbit
    cstate = f'{ctrl_state:b}'.rjust(num_ctrl_qubits, '0')[::-1]
    for i in range(num_ctrl_qubits):
        fc_open_close = ec if cstate[i] == '1' else self._style['bg']
        text_top = None
        if text:
            if top and qlist[i] == min_ctbit:
                text_top = True
            elif not top and qlist[i] == max_ctbit:
                text_top = False
        self._ctrl_qubit(qbit[i], glob_data, fc=fc_open_close, ec=ec, tc=tc, text=text, text_top=text_top)