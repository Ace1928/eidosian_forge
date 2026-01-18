import dataclasses
import math
from typing import Iterable, Callable
from qiskit.circuit import (
from qiskit._qasm2 import (  # pylint: disable=no-name-in-module
from .exceptions import QASM2ParseError
def _gate_builder(name, num_qubits, known_gates, bytecode):
    """Create a gate-builder function of the signature `*params -> Gate` for a gate with a given
    `name`.  This produces a `_DefinedGate` class, whose `_define` method runs through the given
    `bytecode` using the current list of `known_gates` to interpret the gate indices.

    The indirection here is mostly needed to correctly close over `known_gates` and `bytecode`."""

    def definer(*params):
        return _DefinedGate(name, num_qubits, params, known_gates, tuple(bytecode))
    return definer