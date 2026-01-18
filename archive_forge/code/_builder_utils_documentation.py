from __future__ import annotations
import dataclasses
from typing import Iterable, Tuple, Set, Union, TypeVar, TYPE_CHECKING
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.register import Register
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.quantumregister import QuantumRegister

    Ensure that ``true_body`` and ``false_body`` have the same registers defined within them.  These
    do not need to be in the same order between circuits.  The two input circuits are returned,
    mutated to have the same registers.
    