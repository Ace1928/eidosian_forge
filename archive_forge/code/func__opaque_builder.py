import dataclasses
import math
from typing import Iterable, Callable
from qiskit.circuit import (
from qiskit._qasm2 import (  # pylint: disable=no-name-in-module
from .exceptions import QASM2ParseError
def _opaque_builder(name, num_qubits):
    """Create a gate-builder function of the signature `*params -> Gate` for an opaque gate with a
    given `name`, which takes the given numbers of qubits."""

    def definer(*params):
        return Gate(name, num_qubits, params)
    return definer