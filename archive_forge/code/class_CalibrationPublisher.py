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
class CalibrationPublisher(IntEnum):
    """Defines who defined schedule entry."""
    BACKEND_PROVIDER = 0
    QISKIT = 1
    EXPERIMENT_SERVICE = 2