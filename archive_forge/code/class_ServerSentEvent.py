from __future__ import annotations
import json
import inspect
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Iterator, AsyncIterator, cast
from typing_extensions import Self, Protocol, TypeGuard, override, get_origin, runtime_checkable
import httpx
from ._utils import is_mapping, extract_type_var_from_base
from ._exceptions import APIError
class ServerSentEvent:

    def __init__(self, *, event: str | None=None, data: str | None=None, id: str | None=None, retry: int | None=None) -> None:
        if data is None:
            data = ''
        self._id = id
        self._data = data
        self._event = event or None
        self._retry = retry

    @property
    def event(self) -> str | None:
        return self._event

    @property
    def id(self) -> str | None:
        return self._id

    @property
    def retry(self) -> int | None:
        return self._retry

    @property
    def data(self) -> str:
        return self._data

    def json(self) -> Any:
        return json.loads(self.data)

    @override
    def __repr__(self) -> str:
        return f'ServerSentEvent(event={self.event}, data={self.data}, id={self.id}, retry={self.retry})'