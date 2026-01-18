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
class IfElsePlaceholder(InstructionPlaceholder):
    """A placeholder instruction to use in control-flow context managers, when calculating the
    number of resources this instruction should block is deferred until the construction of the
    outer loop.

    This generally should not be instantiated manually; only :obj:`.IfContext` and
    :obj:`.ElseContext` should do it when they need to defer creation of the concrete instruction.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """

    def __init__(self, condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr, true_block: ControlFlowBuilderBlock, false_block: ControlFlowBuilderBlock | None=None, *, label: Optional[str]=None):
        """
        Args:
            condition: the condition to execute the true block on.  This has the same semantics as
                the ``condition`` argument to :obj:`.IfElseOp`.
            true_block: the unbuilt scope block that will become the "true" branch at creation time.
            false_block: if given, the unbuilt scope block that will become the "false" branch at
                creation time.
            label: the label to give the operator when it is created.
        """
        self.__true_block = true_block
        self.__false_block: Optional[ControlFlowBuilderBlock] = false_block
        self.__resources = self._calculate_placeholder_resources()
        super().__init__('if_else', len(self.__resources.qubits), len(self.__resources.clbits), [], label=label)
        self.condition = validate_condition(condition)

    def with_false_block(self, false_block: ControlFlowBuilderBlock) -> 'IfElsePlaceholder':
        """Return a new placeholder instruction, with the false block set to the given value,
        updating the bits used by both it and the true body, if necessary.

        It is an error to try and set the false block on a placeholder that already has one.

        Args:
            false_block: The (unbuilt) instruction scope to set the false body to.

        Returns:
            A new placeholder, with ``false_block`` set to the given input, and both true and false
            blocks expanded to account for all resources.

        Raises:
            CircuitError: if the false block of this placeholder instruction is already set.
        """
        if self.__false_block is not None:
            raise CircuitError(f'false block is already set to {self.__false_block}')
        true_block = self.__true_block.copy()
        true_bits = true_block.qubits() | true_block.clbits()
        false_bits = false_block.qubits() | false_block.clbits()
        true_block.add_bits(false_bits - true_bits)
        false_block.add_bits(true_bits - false_bits)
        return type(self)(self.condition, true_block, false_block, label=self.label)

    def registers(self):
        """Get the registers used by the interior blocks."""
        if self.__false_block is None:
            return self.__true_block.registers.copy()
        return self.__true_block.registers | self.__false_block.registers

    def _calculate_placeholder_resources(self) -> InstructionResources:
        """Get the placeholder resources (see :meth:`.placeholder_resources`).

        This is a separate function because we use the resources during the initialisation to
        determine how we should set our ``num_qubits`` and ``num_clbits``, so we implement the
        public version as a cache access for efficiency.
        """
        if self.__false_block is None:
            qregs, cregs = partition_registers(self.__true_block.registers)
            return InstructionResources(qubits=tuple(self.__true_block.qubits()), clbits=tuple(self.__true_block.clbits()), qregs=tuple(qregs), cregs=tuple(cregs))
        true_qregs, true_cregs = partition_registers(self.__true_block.registers)
        false_qregs, false_cregs = partition_registers(self.__false_block.registers)
        return InstructionResources(qubits=tuple(self.__true_block.qubits() | self.__false_block.qubits()), clbits=tuple(self.__true_block.clbits() | self.__false_block.clbits()), qregs=tuple(true_qregs) + tuple(false_qregs), cregs=tuple(true_cregs) + tuple(false_cregs))

    def placeholder_resources(self):
        return self.__resources

    def concrete_instruction(self, qubits, clbits):
        current_qubits = self.__true_block.qubits()
        current_clbits = self.__true_block.clbits()
        if self.__false_block is not None:
            current_qubits = current_qubits | self.__false_block.qubits()
            current_clbits = current_clbits | self.__false_block.clbits()
        all_bits = qubits | clbits
        current_bits = current_qubits | current_clbits
        if current_bits - all_bits:
            raise CircuitError(f'This block contains bits that are not in the operands sets: {current_bits - all_bits!r}')
        true_body = self.__true_block.build(qubits, clbits)
        if self.__false_block is None:
            false_body = None
        else:
            true_body, false_body = unify_circuit_resources((true_body, self.__false_block.build(qubits, clbits)))
        return (self._copy_mutable_properties(IfElseOp(self.condition, true_body, false_body, label=self.label)), InstructionResources(qubits=tuple(true_body.qubits), clbits=tuple(true_body.clbits), qregs=tuple(true_body.qregs), cregs=tuple(true_body.cregs)))

    def c_if(self, classical, val):
        raise NotImplementedError('IfElseOp cannot be classically controlled through Instruction.c_if. Please nest it in another IfElseOp instead.')