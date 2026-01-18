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