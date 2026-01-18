from dataclasses import asdict, dataclass, field
from typing import Type
from typing_extensions import override
@dataclass
class _Progress(_BaseProgress):
    """Track aggregated and current progress.

    Args:
        total: Intended to track the total progress of an event.
        current: Intended to track the current progress of an event.

    """
    total: _ReadyCompletedTracker = field(default_factory=_ProcessedTracker)
    current: _ReadyCompletedTracker = field(default_factory=_ProcessedTracker)

    def __post_init__(self) -> None:
        if self.total.__class__ is not self.current.__class__:
            raise ValueError('The `total` and `current` instances should be of the same class')

    def increment_ready(self) -> None:
        self.total.ready += 1
        self.current.ready += 1

    def increment_started(self) -> None:
        if not isinstance(self.total, _StartedTracker):
            raise TypeError(f"`{self.total.__class__.__name__}` doesn't have a `started` attribute")
        self.total.started += 1
        self.current.started += 1

    def increment_processed(self) -> None:
        if not isinstance(self.total, _ProcessedTracker):
            raise TypeError(f"`{self.total.__class__.__name__}` doesn't have a `processed` attribute")
        self.total.processed += 1
        self.current.processed += 1

    def increment_completed(self) -> None:
        self.total.completed += 1
        self.current.completed += 1

    @classmethod
    def from_defaults(cls, tracker_cls: Type[_ReadyCompletedTracker], **kwargs: int) -> '_Progress':
        """Utility function to easily create an instance from keyword arguments to both ``Tracker``s."""
        return cls(total=tracker_cls(**kwargs), current=tracker_cls(**kwargs))

    @override
    def reset(self) -> None:
        self.total.reset()
        self.current.reset()

    def reset_on_run(self) -> None:
        self.current.reset()

    def reset_on_restart(self) -> None:
        self.current.reset_on_restart()

    @override
    def load_state_dict(self, state_dict: dict) -> None:
        self.total.load_state_dict(state_dict['total'])
        self.current.load_state_dict(state_dict['current'])