from __future__ import annotations
import copy
import multiprocessing as mp
import typing
from collections import OrderedDict, defaultdict, namedtuple
from typing import (
import numpy as np
from qiskit._accelerate.quantum_circuit import CircuitData
from qiskit.exceptions import QiskitError
from qiskit.utils.multiprocessing import is_main_process
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from . import _classical_resource_map
from ._utils import sort_parameters
from .controlflow.builder import CircuitScopeInterface, ControlFlowBuilderBlock
from .controlflow.break_loop import BreakLoopOp, BreakLoopPlaceholder
from .controlflow.continue_loop import ContinueLoopOp, ContinueLoopPlaceholder
from .controlflow.for_loop import ForLoopOp, ForLoopContext
from .controlflow.if_else import IfElseOp, IfContext
from .controlflow.switch_case import SwitchCaseOp, SwitchContext
from .controlflow.while_loop import WhileLoopOp, WhileLoopContext
from .classical import expr
from .parameterexpression import ParameterExpression, ParameterValueType
from .quantumregister import QuantumRegister, Qubit, AncillaRegister, AncillaQubit
from .classicalregister import ClassicalRegister, Clbit
from .parametertable import ParameterReferences, ParameterTable, ParameterView
from .parametervector import ParameterVector
from .instructionset import InstructionSet
from .operation import Operation
from .register import Register
from .bit import Bit
from .quantumcircuitdata import QuantumCircuitData, CircuitInstruction
from .delay import Delay
def _bit_argument_conversion(specifier, bit_sequence, bit_set, type_) -> list[Bit]:
    """Get the list of bits referred to by the specifier ``specifier``.

    Valid types for ``specifier`` are integers, bits of the correct type (as given in ``type_``), or
    iterables of one of those two scalar types.  Integers are interpreted as indices into the
    sequence ``bit_sequence``.  All allowed bits must be in ``bit_set`` (which should implement
    fast lookup), which is assumed to contain the same bits as ``bit_sequence``.

    Returns:
        List[Bit]: a list of the specified bits from ``bits``.

    Raises:
        CircuitError: if an incorrect type or index is encountered, if the same bit is specified
            more than once, or if the specifier is to a bit not in the ``bit_set``.
    """
    if isinstance(specifier, type_):
        if specifier in bit_set:
            return [specifier]
        raise CircuitError(f"Bit '{specifier}' is not in the circuit.")
    if isinstance(specifier, (int, np.integer)):
        try:
            return [bit_sequence[specifier]]
        except IndexError as ex:
            raise CircuitError(f'Index {specifier} out of range for size {len(bit_sequence)}.') from ex
    if isinstance(specifier, slice):
        return bit_sequence[specifier]
    try:
        return [_bit_argument_conversion_scalar(index, bit_sequence, bit_set, type_) for index in specifier]
    except TypeError as ex:
        message = f"Incorrect bit type: expected '{type_.__name__}' but got '{type(specifier).__name__}'" if isinstance(specifier, Bit) else f"Invalid bit index: '{specifier}' of type '{type(specifier)}'"
        raise CircuitError(message) from ex