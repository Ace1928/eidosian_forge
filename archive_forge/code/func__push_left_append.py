from __future__ import annotations
import abc
from typing import Callable, Tuple
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleComponent
from qiskit.pulse.utils import instruction_duration_validation
@staticmethod
def _push_left_append(this: Schedule, other: ScheduleComponent) -> Schedule:
    """Return ``this`` with ``other`` inserted at the maximum time over
        all channels shared between ```this`` and ``other``.

        Args:
            this: Input schedule to which ``other`` will be inserted.
            other: Other schedule to insert.

        Returns:
            Push left appended schedule.
        """
    this_channels = set(this.channels)
    other_channels = set(other.channels)
    shared_channels = list(this_channels & other_channels)
    ch_slacks = [this.stop_time - this.ch_stop_time(channel) + other.ch_start_time(channel) for channel in shared_channels]
    if ch_slacks:
        slack_chan = shared_channels[np.argmin(ch_slacks)]
        shared_insert_time = this.ch_stop_time(slack_chan) - other.ch_start_time(slack_chan)
    else:
        shared_insert_time = 0
    other_only_insert_time = other.ch_start_time(*other_channels - this_channels)
    insert_time = max(shared_insert_time, other_only_insert_time)
    return this.insert(insert_time, other, inplace=True)