from __future__ import annotations
import contextvars
import functools
import itertools
import sys
import uuid
import warnings
from collections.abc import Generator, Callable, Iterable
from contextlib import contextmanager
from functools import singledispatchmethod
from typing import TypeVar, ContextManager, TypedDict, Union, Optional, Dict
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse import (
from qiskit.providers.backend import BackendV2
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.alignments import AlignmentKind
class _PulseBuilder:
    """Builder context class."""
    __alignment_kinds__ = {'left': transforms.AlignLeft(), 'right': transforms.AlignRight(), 'sequential': transforms.AlignSequential()}

    def __init__(self, backend=None, block: ScheduleBlock | None=None, name: str | None=None, default_alignment: str | AlignmentKind='left'):
        """Initialize the builder context.

        .. note::
            At some point we may consider incorporating the builder into
            the :class:`~qiskit.pulse.Schedule` class. However, the risk of
            this is tying the user interface to the intermediate
            representation. For now we avoid this at the cost of some code
            duplication.

        Args:
            backend (Backend): Input backend to use in
                builder. If not set certain functionality will be unavailable.
            block: Initital ``ScheduleBlock`` to build on.
            name: Name of pulse program to be built.
            default_alignment: Default scheduling alignment for builder.
                One of ``left``, ``right``, ``sequential`` or an instance of
                :class:`~qiskit.pulse.transforms.alignments.AlignmentKind` subclass.

        Raises:
            PulseError: When invalid ``default_alignment`` or `block` is specified.
        """
        self._backend = backend
        self._backend_ctx_token: contextvars.Token[_PulseBuilder] | None = None
        self._context_stack: list[ScheduleBlock] = []
        self._name = name
        if block is not None:
            if isinstance(block, ScheduleBlock):
                root_block = block
            elif isinstance(block, Schedule):
                root_block = self._naive_typecast_schedule(block)
            else:
                raise exceptions.PulseError(f'Input `block` type {block.__class__.__name__} is not a valid format. Specify a pulse program.')
            self._context_stack.append(root_block)
        alignment = _PulseBuilder.__alignment_kinds__.get(default_alignment, default_alignment)
        if not isinstance(alignment, AlignmentKind):
            raise exceptions.PulseError(f'Given `default_alignment` {repr(default_alignment)} is not a valid transformation. Set one of {', '.join(_PulseBuilder.__alignment_kinds__.keys())}, or set an instance of `AlignmentKind` subclass.')
        self.push_context(alignment)

    def __enter__(self) -> ScheduleBlock:
        """Enter this builder context and yield either the supplied schedule
        or the schedule created for the user.

        Returns:
            The schedule that the builder will build on.
        """
        self._backend_ctx_token = BUILDER_CONTEXTVAR.set(self)
        output = self._context_stack[0]
        output._name = self._name or output.name
        return output

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the builder context and compile the built pulse program."""
        self.compile()
        BUILDER_CONTEXTVAR.reset(self._backend_ctx_token)

    @property
    def backend(self):
        """Returns the builder backend if set.

        Returns:
            Optional[Backend]: The builder's backend.
        """
        return self._backend

    def push_context(self, alignment: AlignmentKind):
        """Push new context to the stack."""
        self._context_stack.append(ScheduleBlock(alignment_context=alignment))

    def pop_context(self) -> ScheduleBlock:
        """Pop the last context from the stack."""
        if len(self._context_stack) == 1:
            raise exceptions.PulseError('The root context cannot be popped out.')
        return self._context_stack.pop()

    def get_context(self) -> ScheduleBlock:
        """Get current context.

        Notes:
            New instruction can be added by `.append_subroutine` or `.append_instruction` method.
            Use above methods rather than directly accessing to the current context.
        """
        return self._context_stack[-1]

    @property
    @_requires_backend
    def num_qubits(self):
        """Get the number of qubits in the backend."""
        if isinstance(self.backend, BackendV2):
            return self.backend.num_qubits
        return self.backend.configuration().n_qubits

    def compile(self) -> ScheduleBlock:
        """Compile and output the built pulse program."""
        while len(self._context_stack) > 1:
            current = self.pop_context()
            self.append_subroutine(current)
        return self._context_stack[0]

    def append_instruction(self, instruction: instructions.Instruction):
        """Add an instruction to the builder's context schedule.

        Args:
            instruction: Instruction to append.
        """
        self._context_stack[-1].append(instruction)

    def append_reference(self, name: str, *extra_keys: str):
        """Add external program as a :class:`~qiskit.pulse.instructions.Reference` instruction.

        Args:
            name: Name of subroutine.
            extra_keys: Assistance keys to uniquely specify the subroutine.
        """
        inst = instructions.Reference(name, *extra_keys)
        self.append_instruction(inst)

    def append_subroutine(self, subroutine: Schedule | ScheduleBlock):
        """Append a :class:`ScheduleBlock` to the builder's context schedule.

        This operation doesn't create a reference. Subroutine is directly
        appended to current context schedule.

        Args:
            subroutine: ScheduleBlock to append to the current context block.

        Raises:
            PulseError: When subroutine is not Schedule nor ScheduleBlock.
        """
        if not isinstance(subroutine, (ScheduleBlock, Schedule)):
            raise exceptions.PulseError(f"'{subroutine.__class__.__name__}' is not valid data format in the builder. 'Schedule' and 'ScheduleBlock' can be appended to the builder context.")
        if len(subroutine) == 0:
            return
        if isinstance(subroutine, Schedule):
            subroutine = self._naive_typecast_schedule(subroutine)
        self._context_stack[-1].append(subroutine)

    @singledispatchmethod
    def call_subroutine(self, subroutine: Schedule | ScheduleBlock, name: str | None=None, value_dict: dict[ParameterExpression, ParameterValueType] | None=None, **kw_params: ParameterValueType):
        """Call a schedule or circuit defined outside of the current scope.

        The ``subroutine`` is appended to the context schedule as a call instruction.
        This logic just generates a convenient program representation in the compiler.
        Thus, this doesn't affect execution of inline subroutines.
        See :class:`~pulse.instructions.Call` for more details.

        Args:
            subroutine: Target schedule or circuit to append to the current context.
            name: Name of subroutine if defined.
            value_dict: Parameter object and assigned value mapping. This is more precise way to
                identify a parameter since mapping is managed with unique object id rather than
                name. Especially there is any name collision in a parameter table.
            kw_params: Parameter values to bind to the target subroutine
                with string parameter names. If there are parameter name overlapping,
                these parameters are updated with the same assigned value.

        Raises:
            PulseError:
                - When input subroutine is not valid data format.
        """
        raise exceptions.PulseError(f'Subroutine type {subroutine.__class__.__name__} is not valid data format. Call Schedule, or ScheduleBlock.')

    @call_subroutine.register
    def _(self, target_block: ScheduleBlock, name: Optional[str]=None, value_dict: Optional[Dict[ParameterExpression, ParameterValueType]]=None, **kw_params: ParameterValueType):
        if len(target_block) == 0:
            return
        local_assignment = {}
        for param_name, value in kw_params.items():
            params = target_block.get_parameters(param_name)
            if not params:
                raise exceptions.PulseError(f'Parameter {param_name} is not defined in the target subroutine. {', '.join(map(str, target_block.parameters))} can be specified.')
            for param in params:
                local_assignment[param] = value
        if value_dict:
            if local_assignment.keys() & value_dict.keys():
                warnings.warn("Some parameters provided by 'value_dict' conflict with one through keyword arguments. Parameter values in the keyword arguments are overridden by the dictionary values.", UserWarning)
            local_assignment.update(value_dict)
        if local_assignment:
            target_block = target_block.assign_parameters(local_assignment, inplace=False)
        if name is None:
            keys: tuple[str, ...] = (target_block.name, uuid.uuid4().hex)
        else:
            keys = (name,)
        self.append_reference(*keys)
        self.get_context().assign_references({keys: target_block}, inplace=True)

    @call_subroutine.register
    def _(self, target_schedule: Schedule, name: Optional[str]=None, value_dict: Optional[Dict[ParameterExpression, ParameterValueType]]=None, **kw_params: ParameterValueType):
        if len(target_schedule) == 0:
            return
        self.call_subroutine(self._naive_typecast_schedule(target_schedule), name=name, value_dict=value_dict, **kw_params)

    @staticmethod
    def _naive_typecast_schedule(schedule: Schedule):
        from qiskit.pulse.transforms import inline_subroutines, flatten, pad
        preprocessed_schedule = inline_subroutines(flatten(schedule))
        pad(preprocessed_schedule, inplace=True, pad_with=instructions.TimeBlockade)
        target_block = ScheduleBlock(name=schedule.name)
        for _, inst in preprocessed_schedule.instructions:
            target_block.append(inst, inplace=True)
        return target_block

    def get_dt(self):
        """Retrieve dt differently based on the type of Backend"""
        if isinstance(self.backend, BackendV2):
            return self.backend.dt
        return self.backend.configuration().dt