from __future__ import annotations
import functools
import warnings
from collections import defaultdict
from collections.abc import Iterable, Callable
from qiskit import circuit
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.calibration_entries import (
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleBlock
def _get_instruction_string(inst: str | circuit.instruction.Instruction) -> str:
    if isinstance(inst, str):
        return inst
    else:
        try:
            return inst.name
        except AttributeError as ex:
            raise PulseError('Input "inst" has no attribute "name". This should be a circuit "Instruction".') from ex