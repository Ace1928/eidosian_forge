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
def _remove_timeslots(self, time: int, schedule: 'ScheduleComponent'):
    """Delete the timeslots if present for the respective schedule component.

        Args:
            time: The time to remove the timeslots for the ``schedule`` component.
            schedule: The schedule to insert into self.

        Raises:
            PulseError: If timeslots overlap or an invalid start time is provided.
        """
    if not isinstance(time, int):
        raise PulseError('Schedule start time must be an integer.')
    for channel in schedule.channels:
        if channel not in self._timeslots:
            raise PulseError(f'The channel {channel} is not present in the schedule')
        channel_timeslots = self._timeslots[channel]
        other_timeslots = _get_timeslots(schedule)
        for interval in other_timeslots[channel]:
            if channel_timeslots:
                interval = (interval[0] + time, interval[1] + time)
                index = _interval_index(channel_timeslots, interval)
                if channel_timeslots[index] == interval:
                    channel_timeslots.pop(index)
                    continue
            raise PulseError("Cannot find interval ({t0}, {tf}) to remove from channel {ch} in Schedule(name='{name}').".format(ch=channel, t0=interval[0], tf=interval[1], name=schedule.name))
        if not channel_timeslots:
            self._timeslots.pop(channel)