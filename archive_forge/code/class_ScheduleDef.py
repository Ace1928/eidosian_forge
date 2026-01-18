from __future__ import annotations
import inspect
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence, Callable
from enum import IntEnum
from typing import Any
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.qobj.pulse_qobj import PulseQobjInstruction
from qiskit.exceptions import QiskitError
class ScheduleDef(CalibrationEntry):
    """In-memory Qiskit Pulse representation.

    A pulse schedule must provide signature with the .parameters attribute.
    This entry can be parameterized by a Qiskit Parameter object.
    The .get_schedule method returns a parameter-assigned pulse program.

    .. see_also::
        :class:`.CalibrationEntry` for the purpose of this class.

    """

    def __init__(self, arguments: Sequence[str] | None=None):
        """Define an empty entry.

        Args:
            arguments: User provided argument names for this entry, if parameterized.

        Raises:
            PulseError: When `arguments` is not a sequence of string.
        """
        if arguments and (not all((isinstance(arg, str) for arg in arguments))):
            raise PulseError(f'Arguments must be name of parameters. Not {arguments}.')
        if arguments:
            arguments = list(arguments)
        self._user_arguments = arguments
        self._definition: Callable | Schedule | None = None
        self._signature: inspect.Signature | None = None
        self._user_provided: bool | None = None

    @property
    def user_provided(self) -> bool:
        return self._user_provided

    def _parse_argument(self):
        """Generate signature from program and user provided argument names."""
        all_argnames = {x.name for x in self._definition.parameters}
        if self._user_arguments:
            if set(self._user_arguments) != all_argnames:
                raise PulseError(f"Specified arguments don't match with schedule parameters. {self._user_arguments} != {self._definition.parameters}.")
            argnames = list(self._user_arguments)
        else:
            argnames = sorted(all_argnames)
        params = []
        for argname in argnames:
            param = inspect.Parameter(argname, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
            params.append(param)
        signature = inspect.Signature(parameters=params, return_annotation=type(self._definition))
        self._signature = signature

    def define(self, definition: Schedule | ScheduleBlock, user_provided: bool=True):
        self._definition = definition
        self._parse_argument()
        self._user_provided = user_provided

    def get_signature(self) -> inspect.Signature:
        return self._signature

    def get_schedule(self, *args, **kwargs) -> Schedule | ScheduleBlock:
        if not args and (not kwargs):
            out = self._definition
        else:
            try:
                to_bind = self.get_signature().bind_partial(*args, **kwargs)
            except TypeError as ex:
                raise PulseError("Assigned parameter doesn't match with schedule parameters.") from ex
            value_dict = {}
            for param in self._definition.parameters:
                try:
                    value_dict[param] = to_bind.arguments[param.name]
                except KeyError:
                    pass
            out = self._definition.assign_parameters(value_dict, inplace=False)
        if 'publisher' not in out.metadata:
            if self.user_provided:
                out.metadata['publisher'] = CalibrationPublisher.QISKIT
            else:
                out.metadata['publisher'] = CalibrationPublisher.BACKEND_PROVIDER
        return out

    def __eq__(self, other):
        if hasattr(other, '_definition'):
            return self._definition == other._definition
        return False

    def __str__(self):
        out = f'Schedule {self._definition.name}'
        params_str = ', '.join(self.get_signature().parameters.keys())
        if params_str:
            out += f'({params_str})'
        return out