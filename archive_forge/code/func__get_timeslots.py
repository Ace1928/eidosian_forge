from __future__ import annotations
import abc
import copy
import functools
import itertools
import multiprocessing as mp
import sys
import warnings
from collections.abc import Callable, Iterable
from typing import List, Tuple, Union, Dict, Any
import numpy as np
import rustworkx as rx
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError, UnassignedReferenceError
from qiskit.pulse.instructions import Instruction, Reference
from qiskit.pulse.utils import instruction_duration_validation
from qiskit.pulse.reference_manager import ReferenceManager
from qiskit.utils.multiprocessing import is_main_process
def _get_timeslots(schedule: 'ScheduleComponent') -> TimeSlots:
    """Generate timeslots from given schedule component.

    Args:
        schedule: Input schedule component.

    Raises:
        PulseError: When invalid schedule type is specified.
    """
    if isinstance(schedule, Instruction):
        duration = schedule.duration
        instruction_duration_validation(duration)
        timeslots = {channel: [(0, duration)] for channel in schedule.channels}
    elif isinstance(schedule, Schedule):
        timeslots = schedule.timeslots
    else:
        raise PulseError(f'Invalid schedule type {type(schedule)} is specified.')
    return timeslots