from __future__ import annotations
import typing
from collections.abc import Callable
from numpy import pi
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RYGate, RZGate, CXGate
from .two_local import TwoLocal
Return the parameter bounds.

        Returns:
            The parameter bounds.
        