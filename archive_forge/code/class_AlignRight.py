from __future__ import annotations
import abc
from typing import Callable, Tuple
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleComponent
from qiskit.pulse.utils import instruction_duration_validation
class AlignRight(AlignmentKind):
    """Align instructions in as-late-as-possible manner.

    Instructions are placed at latest available timeslots.
    """

    def __init__(self):
        """Create new right-justified context."""
        super().__init__(context_params=())

    @property
    def is_sequential(self) -> bool:
        return False

    def align(self, schedule: Schedule) -> Schedule:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.

        Returns:
            Schedule with reallocated instructions.
        """
        aligned = Schedule.initialize_from(schedule)
        for _, child in reversed(schedule.children):
            aligned = self._push_right_prepend(aligned, child)
        return aligned

    @staticmethod
    def _push_right_prepend(this: Schedule, other: ScheduleComponent) -> Schedule:
        """Return ``this`` with ``other`` inserted at the latest possible time
        such that ``other`` ends before it overlaps with any of ``this``.

        If required ``this`` is shifted  to start late enough so that there is room
        to insert ``other``.

        Args:
           this: Input schedule to which ``other`` will be inserted.
           other: Other schedule to insert.

        Returns:
           Push right prepended schedule.
        """
        this_channels = set(this.channels)
        other_channels = set(other.channels)
        shared_channels = list(this_channels & other_channels)
        ch_slacks = [this.ch_start_time(channel) - other.ch_stop_time(channel) for channel in shared_channels]
        if ch_slacks:
            insert_time = min(ch_slacks) + other.start_time
        else:
            insert_time = this.stop_time - other.stop_time + other.start_time
        if insert_time < 0:
            this.shift(-insert_time, inplace=True)
            this.insert(0, other, inplace=True)
        else:
            this.insert(insert_time, other, inplace=True)
        return this