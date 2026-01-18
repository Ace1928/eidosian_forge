from __future__ import annotations
import abc
from typing import Callable, Tuple
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleComponent
from qiskit.pulse.utils import instruction_duration_validation
class AlignSequential(AlignmentKind):
    """Align instructions sequentially.

    Instructions played on different channels are also arranged in a sequence.
    No buffer time is inserted in between instructions.
    """

    def __init__(self):
        """Create new sequential context."""
        super().__init__(context_params=())

    @property
    def is_sequential(self) -> bool:
        return True

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
        for _, child in schedule.children:
            aligned.insert(aligned.duration, child, inplace=True)
        return aligned