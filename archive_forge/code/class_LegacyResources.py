from __future__ import annotations
import dataclasses
from typing import Iterable, Tuple, Set, Union, TypeVar, TYPE_CHECKING
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.register import Register
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.quantumregister import QuantumRegister
@dataclasses.dataclass
class LegacyResources:
    """A pair of the :class:`.Clbit` and :class:`.ClassicalRegister` resources used by some other
    object (such as a legacy condition or :class:`.expr.Expr` node)."""
    clbits: tuple[Clbit, ...]
    cregs: tuple[ClassicalRegister, ...]