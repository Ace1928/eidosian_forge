from dataclasses import asdict, dataclass, field
from typing import Type
from typing_extensions import override
@dataclass
class _BaseProgress:
    """Mixin that implements state-loading utilities for dataclasses."""

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> '_BaseProgress':
        obj = cls()
        obj.load_state_dict(state_dict)
        return obj

    def reset(self) -> None:
        """Reset the object's state."""
        raise NotImplementedError