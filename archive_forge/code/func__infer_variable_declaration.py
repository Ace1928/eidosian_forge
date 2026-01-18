import collections
import re
import io
import itertools
import numbers
from os.path import dirname, join, abspath
from typing import Iterable, List, Sequence, Union
from qiskit.circuit import (
from qiskit.circuit.bit import Bit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import (
from qiskit.circuit.library import standard_gates
from qiskit.circuit.register import Register
from qiskit.circuit.tools import pi_check
from . import ast
from .experimental import ExperimentalFeatures
from .exceptions import QASM3ExporterError
from .printer import BasicPrinter
def _infer_variable_declaration(circuit: QuantumCircuit, parameter: Parameter, parameter_name: ast.Identifier) -> Union[ast.ClassicalDeclaration, None]:
    """Attempt to infer what type a parameter should be declared as to work with a circuit.

    This is very simplistic; it assumes all parameters are real numbers that need to be input to the
    program, unless one is used as a loop variable, in which case it shouldn't be declared at all,
    because the ``for`` loop declares it implicitly (per the Qiskit/QSS reading of the OpenQASM
    spec at Qiskit/openqasm@8ee55ec).

    .. note::

        This is a hack around not having a proper type system implemented in Terra, and really this
        whole function should be removed in favour of proper symbol-table building and lookups.
        This function is purely to try and hack the parameters for ``for`` loops into the exporter
        for now.

    Args:
        circuit: The global-scope circuit, which is the base of the exported program.
        parameter: The parameter to infer the type of.
        parameter_name: The name of the parameter to use in the declaration.

    Returns:
        A suitable :obj:`.ast.ClassicalDeclaration` node, or, if the parameter should *not* be
        declared, then ``None``.
    """

    def is_loop_variable(circuit, parameter):
        """Recurse into the instructions a parameter is used in, checking at every level if it is
        used as the loop variable of a ``for`` loop."""
        for instruction, index in circuit._parameter_table[parameter]:
            if isinstance(instruction, ForLoopOp):
                if index == 1:
                    return True
            if isinstance(instruction, ControlFlowOp):
                if is_loop_variable(instruction.params[index], parameter):
                    return True
        return False
    if is_loop_variable(circuit, parameter):
        return None
    return ast.IODeclaration(ast.IOModifier.INPUT, ast.FloatType.DOUBLE, parameter_name)