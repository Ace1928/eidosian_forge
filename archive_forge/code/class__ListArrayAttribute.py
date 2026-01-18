import dataclasses
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
class _ListArrayAttribute(GitlabAttribute):
    """Helper class to support `list` / `array` types."""

    def set_from_cli(self, cli_value: str) -> None:
        if not cli_value.strip():
            self._value = []
        else:
            self._value = [item.strip() for item in cli_value.split(',')]

    def get_for_api(self, *, key: str) -> Tuple[str, str]:
        if isinstance(self._value, str):
            return (key, self._value)
        if TYPE_CHECKING:
            assert isinstance(self._value, list)
        return (key, ','.join([str(x) for x in self._value]))