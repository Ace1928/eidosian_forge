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
def _lookup_variable(self, variable) -> ast.Identifier:
    """Lookup a Terra object within the current context, and return the name that should be used
        to represent it in OpenQASM 3 programmes."""
    if isinstance(variable, Bit):
        variable = self.current_scope().bit_map[variable]
    for scope in reversed(self.current_context()):
        if variable in scope.symbol_map:
            return scope.symbol_map[variable]
    raise KeyError(f"'{variable}' is not defined in the current context")