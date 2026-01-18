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
def add_calibration(self, gate: Union[Gate, str], qubits: Sequence[int], schedule, params: Sequence[ParameterValueType] | None=None) -> None:
    """Register a low-level, custom pulse definition for the given gate.

        Args:
            gate (Union[Gate, str]): Gate information.
            qubits (Union[int, Tuple[int]]): List of qubits to be measured.
            schedule (Schedule): Schedule information.
            params (Optional[List[Union[float, Parameter]]]): A list of parameters.

        Raises:
            Exception: if the gate is of type string and params is None.
        """

    def _format(operand):
        try:
            evaluated = complex(operand)
            if np.isreal(evaluated):
                evaluated = float(evaluated.real)
                if evaluated.is_integer():
                    evaluated = int(evaluated)
            return evaluated
        except TypeError:
            return operand
    if isinstance(gate, Gate):
        params = gate.params
        gate = gate.name
    if params is not None:
        params = tuple(map(_format, params))
    else:
        params = ()
    self._calibrations[gate][tuple(qubits), params] = schedule