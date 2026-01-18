from __future__ import annotations
from datetime import datetime as DateTime
from typing import Any, Callable, Iterator, Mapping, Optional, Union, cast
from constantly import NamedConstant
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.python.failure import Failure
from twisted.python.reflect import safe_repr
from ._flatten import aFormatter, flatFormat
from ._interfaces import LogEvent
class PotentialCallWrapper(object):
    """
    Object wrapper that wraps C{getattr()} so as to process call-parentheses
    C{"()"} after a dotted attribute access.
    """

    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def __getattr__(self, name: str) -> object:
        return keycall(name, self._wrapped.__getattribute__)

    def __getitem__(self, name: str) -> object:
        value = self._wrapped[name]
        return PotentialCallWrapper(value)

    def __format__(self, format_spec: str) -> str:
        return self._wrapped.__format__(format_spec)

    def __repr__(self) -> str:
        return self._wrapped.__repr__()

    def __str__(self) -> str:
        return self._wrapped.__str__()