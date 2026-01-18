from __future__ import annotations
import abc
from typing import Callable, Tuple
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleComponent
from qiskit.pulse.utils import instruction_duration_validation
class AlignmentKind(abc.ABC):
    """An abstract class for schedule alignment."""

    def __init__(self, context_params: Tuple[ParameterValueType, ...]):
        """Create new context."""
        self._context_params = tuple(context_params)

    @abc.abstractmethod
    def align(self, schedule: Schedule) -> Schedule:
        """Reallocate instructions according to the policy.

        Only top-level sub-schedules are aligned. If sub-schedules are nested,
        nested schedules are not recursively aligned.

        Args:
            schedule: Schedule to align.

        Returns:
            Schedule with reallocated instructions.
        """
        pass

    @property
    @abc.abstractmethod
    def is_sequential(self) -> bool:
        """Return ``True`` if this is sequential alignment context.

        This information is used to evaluate DAG equivalency of two :class:`.ScheduleBlock`s.
        When the context has two pulses in different channels,
        a sequential context subtype intends to return following scheduling outcome.

        .. parsed-literal::

                ┌────────┐
            D0: ┤ pulse1 ├────────────
                └────────┘  ┌────────┐
            D1: ────────────┤ pulse2 ├
                            └────────┘

        On the other hand, parallel context with ``is_sequential=False`` returns

        .. parsed-literal::

                ┌────────┐
            D0: ┤ pulse1 ├
                ├────────┤
            D1: ┤ pulse2 ├
                └────────┘

        All subclasses must implement this method according to scheduling strategy.
        """
        pass

    def __eq__(self, other: object) -> bool:
        """Check equality of two transforms."""
        if type(self) is not type(other):
            return False
        if self._context_params != other._context_params:
            return False
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}({', '.join(self._context_params)})'