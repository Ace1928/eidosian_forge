from __future__ import annotations
from datetime import datetime as DateTime
from typing import Any, Callable, Iterator, Mapping, Optional, Union, cast
from constantly import NamedConstant
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.python.failure import Failure
from twisted.python.reflect import safe_repr
from ._flatten import aFormatter, flatFormat
from ._interfaces import LogEvent
class CallMapping(Mapping[str, Any]):
    """
    Read-only mapping that turns a C{()}-suffix in key names into an invocation
    of the key rather than a lookup of the key.

    Implementation support for L{formatWithCall}.
    """

    def __init__(self, submapping: Mapping[str, Any]) -> None:
        """
        @param submapping: Another read-only mapping which will be used to look
            up items.
        """
        self._submapping = submapping

    def __iter__(self) -> Iterator[Any]:
        return iter(self._submapping)

    def __len__(self) -> int:
        return len(self._submapping)

    def __getitem__(self, key: str) -> Any:
        """
        Look up an item in the submapping for this L{CallMapping}, calling it
        if C{key} ends with C{"()"}.
        """
        return keycall(key, self._submapping.__getitem__)