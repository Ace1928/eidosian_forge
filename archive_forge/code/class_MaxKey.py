from __future__ import annotations
from typing import Any
class MaxKey:
    """MongoDB internal MaxKey type."""
    __slots__ = ()
    _type_marker = 127

    def __getstate__(self) -> Any:
        return {}

    def __setstate__(self, state: Any) -> None:
        pass

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, MaxKey)

    def __hash__(self) -> int:
        return hash(self._type_marker)

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __le__(self, other: Any) -> bool:
        return isinstance(other, MaxKey)

    def __lt__(self, dummy: Any) -> bool:
        return False

    def __ge__(self, dummy: Any) -> bool:
        return True

    def __gt__(self, other: Any) -> bool:
        return not isinstance(other, MaxKey)

    def __repr__(self) -> str:
        return 'MaxKey()'