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
class InstructionResources(typing.NamedTuple):
    """The quantum and classical resources used within a particular instruction.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.

    Attributes:
        qubits: A collection of qubits that will be used by the instruction.
        clbits: A collection of clbits that will be used by the instruction.
        qregs: A collection of quantum registers that are used by the instruction.
        cregs: A collection of classical registers that are used by the instruction.
    """
    qubits: Collection[Qubit] = ()
    clbits: Collection[Clbit] = ()
    qregs: Collection[QuantumRegister] = ()
    cregs: Collection[ClassicalRegister] = ()