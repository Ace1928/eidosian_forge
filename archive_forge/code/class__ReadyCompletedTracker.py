from dataclasses import asdict, dataclass, field
from typing import Type
from typing_extensions import override
@dataclass
class _ReadyCompletedTracker(_BaseProgress):
    """Track an event's progress.

    Args:
        ready: Intended to track the number of events ready to start.
        completed: Intended to be incremented after the event completes (e.g. after ``on_*_end`` runs).

    These attributes should be increased in order, that is, :attr:`ready` first and :attr:`completed` last.

    """
    ready: int = 0
    completed: int = 0

    @override
    def reset(self) -> None:
        """Reset the state."""
        self.ready = 0
        self.completed = 0

    def reset_on_restart(self) -> None:
        """Reset the progress on restart.

        If there is a failure before all attributes are increased, restore the attributes to the last fully completed
        value.

        """
        self.ready = self.completed