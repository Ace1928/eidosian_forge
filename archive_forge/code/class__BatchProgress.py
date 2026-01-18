from dataclasses import asdict, dataclass, field
from typing import Type
from typing_extensions import override
@dataclass
class _BatchProgress(_Progress):
    """Tracks batch progress.

    These counters are local to a trainer rank. By default, they are not globally synced across all ranks.

    Args:
        total: Tracks the total batch progress.
        current: Tracks the current batch progress.
        is_last_batch: Whether the batch is the last one. This is useful for iterable datasets.

    """
    is_last_batch: bool = False

    @override
    def reset(self) -> None:
        super().reset()
        self.is_last_batch = False

    @override
    def reset_on_run(self) -> None:
        super().reset_on_run()
        self.is_last_batch = False

    @override
    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.is_last_batch = state_dict['is_last_batch']