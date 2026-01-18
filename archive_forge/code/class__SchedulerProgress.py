from dataclasses import asdict, dataclass, field
from typing import Type
from typing_extensions import override
@dataclass
class _SchedulerProgress(_Progress):
    """Tracks scheduler progress.

    These counters are local to a trainer rank. By default, they are not globally synced across all ranks.

    Args:
        total: Tracks the total scheduler progress.
        current: Tracks the current scheduler progress.

    """
    total: _ReadyCompletedTracker = field(default_factory=_ReadyCompletedTracker)
    current: _ReadyCompletedTracker = field(default_factory=_ReadyCompletedTracker)