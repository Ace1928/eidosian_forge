from dataclasses import asdict, dataclass, field
from typing import Type
from typing_extensions import override
@dataclass
class _ProcessedTracker(_StartedTracker):
    """Track an event's progress.

    Args:
        ready: Intended to track the number of events ready to start.
        started: Intended to be incremented after the event is started (e.g. after ``on_*_start`` runs).
        processed: Intended to be incremented after the event is processed.
        completed: Intended to be incremented after the event completes (e.g. after ``on_*_end`` runs).

    These attributes should be increased in order, that is, :attr:`ready` first and :attr:`completed` last.

    """
    processed: int = 0

    @override
    def reset(self) -> None:
        super().reset()
        self.processed = 0

    @override
    def reset_on_restart(self) -> None:
        super().reset_on_restart()
        self.processed = self.completed