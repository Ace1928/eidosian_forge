from __future__ import annotations
import logging
import os
import subprocess
import tempfile
import shutil
import typing
from warnings import warn
from qiskit import user_config
from qiskit.utils import optionals as _optionals
from qiskit.circuit import ControlFlowOp, Measure
from . import latex as _latex
from . import text as _text
from . import matplotlib as _matplotlib
from . import _utils
from ..utils import _trim as trim_image
from ..exceptions import VisualizationError
def _matplotlib_circuit_drawer(circuit, scale=None, filename=None, style=None, plot_barriers=True, reverse_bits=False, justify=None, idle_wires=True, with_layout=True, fold=None, ax=None, initial_state=False, cregbundle=None, wire_order=None, expr_len=30):
    """Draw a quantum circuit based on matplotlib.
    If `%matplotlib inline` is invoked in a Jupyter notebook, it visualizes a circuit inline.
    We recommend `%config InlineBackend.figure_format = 'svg'` for the inline visualization.

    Args:
        circuit (QuantumCircuit): a quantum circuit
        scale (float): scaling factor
        filename (str): file path to save image to
        style (dict or str): dictionary of style or file name of style file
        reverse_bits (bool): When set to True, reverse the bit order inside
            registers for the output visualization.
        plot_barriers (bool): Enable/disable drawing barriers in the output
            circuit. Defaults to True.
        justify (str): `left`, `right` or `none`. Defaults to `left`. Says how
            the circuit should be justified.
        idle_wires (bool): Include idle wires. Default is True.
        with_layout (bool): Include layout information, with labels on the physical
            layout. Default: True.
        fold (int): Number of vertical layers allowed before folding. Default is 25.
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified, a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.
        initial_state (bool): Optional. Adds |0> in the beginning of the line.
            Default: `False`.
        cregbundle (bool): Optional. If set True bundle classical registers.
            Default: ``True``.
        wire_order (list): Optional. A list of integers used to reorder the display
            of the bits. The list must have an entry for every bit with the bits
            in the range 0 to (num_qubits + num_clbits).
        expr_len (int): Optional. The number of characters to display if an :class:`~.expr.Expr`
            is used for the condition in a :class:`.ControlFlowOp`. If this number is exceeded,
            the string will be truncated at that number and '...' added to the end.

    Returns:
        matplotlib.figure: a matplotlib figure object for the circuit diagram
            if the ``ax`` kwarg is not set.
    """
    qubits, clbits, nodes = _utils._get_layered_instructions(circuit, reverse_bits=reverse_bits, justify=justify, idle_wires=idle_wires, wire_order=wire_order)
    if fold is None:
        fold = 25
    qcd = _matplotlib.MatplotlibDrawer(qubits, clbits, nodes, circuit, scale=scale, style=style, reverse_bits=reverse_bits, plot_barriers=plot_barriers, fold=fold, ax=ax, initial_state=initial_state, cregbundle=cregbundle, with_layout=with_layout, expr_len=expr_len)
    return qcd.draw(filename)