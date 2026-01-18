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
@staticmethod
def _raise_on_jump(operation):
    from .break_loop import BreakLoopOp, BreakLoopPlaceholder
    from .continue_loop import ContinueLoopOp, ContinueLoopPlaceholder
    forbidden = (BreakLoopOp, BreakLoopPlaceholder, ContinueLoopOp, ContinueLoopPlaceholder)
    if isinstance(operation, forbidden):
        raise CircuitError(f"The current builder scope cannot take a '{operation.name}' because it is not in a loop.")