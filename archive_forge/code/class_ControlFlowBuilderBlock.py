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
class ControlFlowBuilderBlock(CircuitScopeInterface):
    """A lightweight scoped block for holding instructions within a control-flow builder context.

    This class is designed only to be used by :obj:`.QuantumCircuit` as an internal context for
    control-flow builder instructions, and in general should never be instantiated by any code other
    than that.

    Note that the instructions that are added to this scope may not be valid yet, so this elides
    some of the type-checking of :obj:`.QuantumCircuit` until those things are known.

    The general principle of the resource tracking through these builder blocks is that every
    necessary resource should pass through an :meth:`.append` call, so that at the point that
    :meth:`.build` is called, the scope knows all the concrete resources that it requires.  However,
    the scope can also contain "placeholder" instructions, which may need extra resources filling in
    from outer scopes (such as a ``break`` needing to know the width of its containing ``for``
    loop).  This means that :meth:`.build` takes all the *containing* scope's resources as well.
    This does not break the "all resources pass through an append" rule, because the containing
    scope will only begin to build its instructions once it has received them all.

    In short, :meth:`.append` adds resources, and :meth:`.build` may use only a subset of the extra
    ones passed.  This ensures that all instructions know about all the resources they need, even in
    the case of ``break``, but do not block any resources that they do *not* need.

    .. warning::

        This is an internal interface and no part of it should be relied upon outside of Qiskit
        Terra.
    """
    __slots__ = ('_instructions', 'registers', 'global_phase', '_allow_jumps', '_parent', '_built', '_forbidden_message')

    def __init__(self, qubits: Iterable[Qubit], clbits: Iterable[Clbit], *, parent: CircuitScopeInterface, registers: Iterable[Register]=(), allow_jumps: bool=True, forbidden_message: Optional[str]=None):
        """
        Args:
            qubits: Any qubits this scope should consider itself as using from the beginning.
            clbits: Any clbits this scope should consider itself as using from the beginning.  Along
                with ``qubits``, this is useful for things such as ``if`` and ``while`` loop
                builders, where the classical condition has associated resources, and is known when
                this scope is created.
            registers: Any registers this scope should consider itself as using from the
                beginning.  This is useful for :obj:`.IfElseOp` and :obj:`.WhileLoopOp` instances
                which use a classical register as their condition.
            allow_jumps: Whether this builder scope should allow ``break`` and ``continue``
                statements within it.  This is intended to help give sensible error messages when
                dangerous behaviour is encountered, such as using ``break`` inside an ``if`` context
                manager that is not within a ``for`` manager.  This can only be safe if the user is
                going to place the resulting :obj:`.QuantumCircuit` inside a :obj:`.ForLoopOp` that
                uses *exactly* the same set of resources.  We cannot verify this from within the
                builder interface (and it is too expensive to do when the ``for`` op is made), so we
                fail safe, and require the user to use the more verbose, internal form.
            parent: The scope interface of the containing scope.
            forbidden_message: If a string is given here, a :exc:`.CircuitError` will be raised on
                any attempts to append instructions to the scope with this message.  This is used by
                pseudo scopes where the state machine of the builder scopes has changed into a
                position where no instructions should be accepted, such as when inside a ``switch``
                but outside any cases.
        """
        self._instructions = CircuitData(qubits, clbits)
        self.registers = set(registers)
        self.global_phase = 0.0
        self._allow_jumps = allow_jumps
        self._parent = parent
        self._built = False
        self._forbidden_message = forbidden_message

    def qubits(self):
        """The set of qubits associated with this scope."""
        return set(self.instructions.qubits)

    def clbits(self):
        """The set of clbits associated with this scope."""
        return set(self.instructions.clbits)

    @property
    def allow_jumps(self):
        """Whether this builder scope should allow ``break`` and ``continue`` statements within it.

        This is intended to help give sensible error messages when dangerous behaviour is
        encountered, such as using ``break`` inside an ``if`` context manager that is not within a
        ``for`` manager.  This can only be safe if the user is going to place the resulting
        :obj:`.QuantumCircuit` inside a :obj:`.ForLoopOp` that uses *exactly* the same set of
        resources.  We cannot verify this from within the builder interface (and it is too expensive
        to do when the ``for`` op is made), so we fail safe, and require the user to use the more
        verbose, internal form.
        """
        return self._allow_jumps

    @property
    def instructions(self):
        return self._instructions

    @staticmethod
    def _raise_on_jump(operation):
        from .break_loop import BreakLoopOp, BreakLoopPlaceholder
        from .continue_loop import ContinueLoopOp, ContinueLoopPlaceholder
        forbidden = (BreakLoopOp, BreakLoopPlaceholder, ContinueLoopOp, ContinueLoopPlaceholder)
        if isinstance(operation, forbidden):
            raise CircuitError(f"The current builder scope cannot take a '{operation.name}' because it is not in a loop.")

    def append(self, instruction: CircuitInstruction) -> CircuitInstruction:
        if self._forbidden_message is not None:
            raise CircuitError(self._forbidden_message)
        if not self._allow_jumps:
            self._raise_on_jump(instruction.operation)
        for b in instruction.qubits:
            self.instructions.add_qubit(b, strict=False)
        for b in instruction.clbits:
            self.instructions.add_clbit(b, strict=False)
        self._instructions.append(instruction)
        return instruction

    def extend(self, data: CircuitData):
        if self._forbidden_message is not None:
            raise CircuitError(self._forbidden_message)
        if not self._allow_jumps:
            data.foreach_op(self._raise_on_jump)
        active_qubits, active_clbits = data.active_bits()
        for b in data.qubits:
            if b in active_qubits:
                self.instructions.add_qubit(b, strict=False)
        for b in data.clbits:
            if b in active_clbits:
                self.instructions.add_clbit(b, strict=False)
        self.instructions.extend(data)

    def resolve_classical_resource(self, specifier):
        if self._built:
            raise CircuitError('Cannot add resources after the scope has been built.')
        resource = self._parent.resolve_classical_resource(specifier)
        if isinstance(resource, Clbit):
            self.add_bits((resource,))
        else:
            self.add_register(resource)
        return resource

    def peek(self) -> CircuitInstruction:
        """Get the value of the most recent instruction tuple in this scope."""
        if not self._instructions:
            raise CircuitError('This scope contains no instructions.')
        return self._instructions[-1]

    def pop(self) -> CircuitInstruction:
        """Get the value of the most recent instruction in this scope, and remove it from this
        object."""
        if not self._instructions:
            raise CircuitError('This scope contains no instructions.')
        return self._instructions.pop()

    def add_bits(self, bits: Iterable[Union[Qubit, Clbit]]):
        """Add extra bits to this scope that are not associated with any concrete instruction yet.

        This is useful for expanding a scope's resource width when it may contain ``break`` or
        ``continue`` statements, or when its width needs to be expanded to match another scope's
        width (as in the case of :obj:`.IfElseOp`).

        Args:
            bits: The qubits and clbits that should be added to a scope.  It is not an error if
                there are duplicates, either within the iterable or with the bits currently in
                scope.

        Raises:
            TypeError: if the provided bit is of an incorrect type.
        """
        for bit in bits:
            if isinstance(bit, Qubit):
                self.instructions.add_qubit(bit, strict=False)
            elif isinstance(bit, Clbit):
                self.instructions.add_clbit(bit, strict=False)
            else:
                raise TypeError(f"Can only add qubits or classical bits, but received '{bit}'.")

    def add_register(self, register: Register):
        """Add a :obj:`.Register` to the set of resources used by this block, ensuring that
        all bits contained within are also accounted for.

        Args:
            register: the register to add to the block.
        """
        if register in self.registers:
            return
        self.registers.add(register)
        self.add_bits(register)

    def build(self, all_qubits: FrozenSet[Qubit], all_clbits: FrozenSet[Clbit]) -> 'qiskit.circuit.QuantumCircuit':
        """Build this scoped block into a complete :obj:`.QuantumCircuit` instance.

        This will build a circuit which contains all of the necessary qubits and clbits and no
        others.

        The ``qubits`` and ``clbits`` arguments should be sets that contains all the resources in
        the outer scope; these will be passed down to inner placeholder instructions, so they can
        apply themselves across the whole scope should they need to.  The resulting
        :obj:`.QuantumCircuit` will be defined over a (nonstrict) subset of these resources.  This
        is used to let ``break`` and ``continue`` span all resources, even if they are nested within
        several :obj:`.IfElsePlaceholder` objects, without requiring :obj:`.IfElsePlaceholder`
        objects *without* any ``break`` or ``continue`` statements to be full-width.

        Args:
            all_qubits: all the qubits in the containing scope of this block.  The block may expand
                to use some or all of these qubits, but will never gain qubits that are not in this
                set.
            all_clbits: all the clbits in the containing scope of this block.  The block may expand
                to use some or all of these clbits, but will never gain clbits that are not in this
                set.

        Returns:
            A circuit containing concrete versions of all the instructions that were in the scope,
            and using the minimal set of resources necessary to support them, within the enclosing
            scope.
        """
        from qiskit.circuit import QuantumCircuit, SwitchCaseOp
        self._built = True
        if self._forbidden_message is not None:
            raise RuntimeError('Cannot build a forbidden scope. Please report this as a bug.')
        potential_qubits = set(all_qubits) - self.qubits()
        potential_clbits = set(all_clbits) - self.clbits()
        out = QuantumCircuit(self._instructions.qubits, self._instructions.clbits, *self.registers, global_phase=self.global_phase)
        placeholder_to_concrete = {}

        def update_registers(index, op):
            if isinstance(op, InstructionPlaceholder):
                op, resources = op.concrete_instruction(all_qubits, all_clbits)
                qubits = tuple(resources.qubits)
                clbits = tuple(resources.clbits)
                placeholder_to_concrete[index] = CircuitInstruction(op, qubits, clbits)
                if potential_qubits and qubits:
                    add_qubits = potential_qubits.intersection(qubits)
                    if add_qubits:
                        potential_qubits.difference_update(add_qubits)
                        out.add_bits(add_qubits)
                if potential_clbits and clbits:
                    add_clbits = potential_clbits.intersection(clbits)
                    if add_clbits:
                        potential_clbits.difference_update(add_clbits)
                        out.add_bits(add_clbits)
                for register in itertools.chain(resources.qregs, resources.cregs):
                    if register not in self.registers:
                        self.add_register(register)
                        out.add_register(register)
            if getattr(op, 'condition', None) is not None:
                for register in condition_resources(op.condition).cregs:
                    if register not in self.registers:
                        self.add_register(register)
                        out.add_register(register)
            elif isinstance(op, SwitchCaseOp):
                target = op.target
                if isinstance(target, Clbit):
                    target_registers = ()
                elif isinstance(target, ClassicalRegister):
                    target_registers = (target,)
                else:
                    target_registers = node_resources(target).cregs
                for register in target_registers:
                    if register not in self.registers:
                        self.add_register(register)
                        out.add_register(register)
        self._instructions.foreach_op_indexed(update_registers)
        out_data = self._instructions.copy()
        out_data.replace_bits(out.qubits, out.clbits)
        for i, instruction in placeholder_to_concrete.items():
            out_data[i] = instruction
        out._current_scope().extend(out_data)
        return out

    def copy(self) -> 'ControlFlowBuilderBlock':
        """Return a semi-shallow copy of this builder block.

        The instruction lists and sets of qubits and clbits will be new instances (so mutations will
        not propagate), but any :obj:`.Instruction` instances within them will not be copied.

        Returns:
            a semi-shallow copy of this object.
        """
        out = type(self).__new__(type(self))
        out._instructions = self._instructions.copy()
        out.registers = self.registers.copy()
        out.global_phase = self.global_phase
        out._parent = self._parent
        out._allow_jumps = self._allow_jumps
        out._forbidden_message = self._forbidden_message
        return out