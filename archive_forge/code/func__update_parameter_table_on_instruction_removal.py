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
def _update_parameter_table_on_instruction_removal(self, instruction: CircuitInstruction):
    """Update the :obj:`.ParameterTable` of this circuit given that an instance of the given
        ``instruction`` has just been removed from the circuit.

        .. note::

            This does not account for the possibility for the same instruction instance being added
            more than once to the circuit.  At the time of writing (2021-11-17, main commit 271a82f)
            there is a defensive ``deepcopy`` of parameterised instructions inside
            :meth:`.QuantumCircuit.append`, so this should be safe.  Trying to account for it would
            involve adding a potentially quadratic-scaling loop to check each entry in ``data``.
        """
    atomic_parameters: list[tuple[Parameter, int]] = []
    for index, parameter in enumerate(instruction.operation.params):
        if isinstance(parameter, (ParameterExpression, QuantumCircuit)):
            atomic_parameters.extend(((p, index) for p in parameter.parameters))
    for atomic_parameter, index in atomic_parameters:
        new_entries = self._parameter_table[atomic_parameter].copy()
        new_entries.discard((instruction.operation, index))
        if not new_entries:
            del self._parameter_table[atomic_parameter]
            self._parameters = None
        else:
            self._parameter_table[atomic_parameter] = new_entries