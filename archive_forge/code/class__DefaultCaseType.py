from __future__ import annotations
import contextlib
from typing import Union, Iterable, Any, Tuple, Optional, List, Literal, TYPE_CHECKING
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from .builder import InstructionPlaceholder, InstructionResources, ControlFlowBuilderBlock
from .control_flow import ControlFlowOp
from ._builder_utils import unify_circuit_resources, partition_registers, node_resources
class _DefaultCaseType:
    """The type of the default-case singleton.  This is used instead of just having
    ``CASE_DEFAULT = object()`` so we can set the pretty-printing properties, which are class-level
    only."""

    def __repr__(self):
        return '<default case>'