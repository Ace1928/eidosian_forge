from __future__ import annotations
import abc
import itertools
import typing
from typing import Collection, Iterable, FrozenSet, Tuple, Union, Optional, Sequence
from qiskit._accelerate.quantum_circuit import CircuitData
from qiskit.circuit.classicalregister import Clbit, ClassicalRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuitdata import CircuitInstruction
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.circuit.register import Register
from ._builder_utils import condition_resources, node_resources
class CircuitScopeInterface(abc.ABC):
    """An interface that circuits and builder blocks explicitly fulfill, which contains the primitive
    methods of circuit construction and object validation.

    This allows core circuit methods to be applied to the currently open builder scope, and allows
    the builders to hook into all places where circuit resources might be used.  This allows the
    builders to track the resources being used, without getting in the way of
    :class:`.QuantumCircuit` doing its own thing.
    """
    __slots__ = ()

    @property
    @abc.abstractmethod
    def instructions(self) -> Sequence[CircuitInstruction]:
        """Indexable view onto the :class:`.CircuitInstruction`s backing this scope."""

    @abc.abstractmethod
    def append(self, instruction: CircuitInstruction) -> CircuitInstruction:
        """Low-level 'append' primitive; this may assume that the qubits, clbits and operation are
        all valid for the circuit.

        Abstraction of :meth:`.QuantumCircuit._append` (the low-level one, not the high-level).

        Args:
            instruction: the resource-validated instruction context object.

        Returns:
            the instruction context object actually appended.  This is not required to be the same
            as the object given (but typically will be).
        """

    @abc.abstractmethod
    def extend(self, data: CircuitData):
        """Appends all instructions from ``data`` to the scope.

        Args:
            data: The instruction listing.
        """

    @abc.abstractmethod
    def resolve_classical_resource(self, specifier: Clbit | ClassicalRegister | int) -> Clbit | ClassicalRegister:
        """Resolve a single bit-like classical-resource specifier.

        A resource refers to either a classical bit or a register, where integers index into the
        classical bits of the greater circuit.

        This is called whenever a classical bit or register is being used outside the standard
        :class:`.Clbit` usage of instructions in :meth:`append`, such as in a legacy two-tuple
        condition.

        Args:
            specifier: the classical resource specifier.

        Returns:
            the resolved resource.  This cannot be an integer any more; an integer input is resolved
            into a classical bit.

        Raises:
            CircuitError: if the resource cannot be used by the scope, such as an out-of-range index
                or a :class:`.Clbit` that isn't actually in the circuit.
        """