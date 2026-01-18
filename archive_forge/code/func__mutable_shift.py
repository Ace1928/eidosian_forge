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
def _mutable_shift(self, time: int) -> 'Schedule':
    """Return this schedule shifted forward by `time`.

        Args:
            time: Time to shift by

        Raises:
            PulseError: if ``time`` is not an integer.
        """
    if not isinstance(time, int):
        raise PulseError('Schedule start time must be an integer.')
    timeslots = {}
    for chan, ch_timeslots in self._timeslots.items():
        timeslots[chan] = [(ts[0] + time, ts[1] + time) for ts in ch_timeslots]
    _check_nonnegative_timeslot(timeslots)
    self._duration = self._duration + time
    self._timeslots = timeslots
    self._children = [(orig_time + time, child) for orig_time, child in self.children]
    return self