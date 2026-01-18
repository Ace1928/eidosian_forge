from __future__ import annotations
from typing import Optional, Union, Iterable, TYPE_CHECKING
import itertools
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.classical import expr
from qiskit.circuit.instructionset import InstructionSet
from qiskit.circuit.exceptions import CircuitError
from .builder import ControlFlowBuilderBlock, InstructionPlaceholder, InstructionResources
from .control_flow import ControlFlowOp
from ._builder_utils import (
class ElseContext:
    """A context manager for building up an ``else`` statements onto circuits in a natural order,
    without having to construct the statement body first.

    Instances of this context manager should only ever be gained as the output of the
    :obj:`.IfContext` manager, so they know what they refer to.  Instances of this context are
    "friends" of the circuit that created the :obj:`.IfContext` that in turn created this object.
    The context will manipulate the circuit's defined scopes when it is entered (by popping the old
    :obj:`.IfElseOp` if it exists and pushing a new scope onto the stack) and exited (by popping its
    scope, building it, and appending the resulting :obj:`.IfElseOp`).

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """
    __slots__ = ('_if_instruction', '_if_registers', '_if_context', '_used')

    def __init__(self, if_context: IfContext):
        self._if_instruction = None
        self._if_registers = None
        self._if_context = if_context
        self._used = False

    def __enter__(self):
        if self._used:
            raise CircuitError("Cannot re-use an 'else' context.")
        self._used = True
        appended_instructions = self._if_context.appended_instructions
        circuit = self._if_context.circuit
        if appended_instructions is None:
            raise CircuitError("Cannot attach an 'else' branch to an incomplete 'if' block.")
        if len(appended_instructions) != 1:
            raise CircuitError("Cannot attach an 'else' to a broadcasted 'if' block.")
        appended = appended_instructions[0]
        instruction = circuit._peek_previous_instruction_in_scope()
        if appended.operation is not instruction.operation:
            raise CircuitError(f"The 'if' block is not the most recent instruction in the circuit. Expected to find: {appended!r}, but instead found: {instruction!r}.")
        self._if_instruction = circuit._pop_previous_instruction_in_scope()
        if isinstance(self._if_instruction.operation, IfElseOp):
            self._if_registers = set(self._if_instruction.operation.blocks[0].cregs).union(self._if_instruction.operation.blocks[0].qregs)
        else:
            self._if_registers = self._if_instruction.operation.registers()
        circuit._push_scope(self._if_instruction.qubits, self._if_instruction.clbits, registers=self._if_registers, allow_jumps=self._if_context.in_loop)

    def __exit__(self, exc_type, exc_val, exc_tb):
        circuit = self._if_context.circuit
        if exc_type is not None:
            circuit._pop_scope()
            circuit._append(self._if_instruction)
            self._used = False
            return False
        false_block = circuit._pop_scope()
        if isinstance(self._if_instruction.operation, IfElsePlaceholder):
            if_operation = self._if_instruction.operation.with_false_block(false_block)
            resources = if_operation.placeholder_resources()
            circuit.append(if_operation, resources.qubits, resources.clbits)
        else:
            true_body = self._if_instruction.operation.blocks[0]
            false_body = false_block.build(false_block.qubits(), false_block.clbits())
            true_body, false_body = unify_circuit_resources((true_body, false_body))
            circuit.append(IfElseOp(self._if_context.condition, true_body, false_body, label=self._if_instruction.operation.label), tuple(true_body.qubits), tuple(true_body.clbits))
        return False