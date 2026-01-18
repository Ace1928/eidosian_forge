from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
class QueryParams(ImmutableMultiDict[str, str]):
    """
    An immutable multidict.
    """

    def __init__(self, *args: ImmutableMultiDict[typing.Any, typing.Any] | typing.Mapping[typing.Any, typing.Any] | list[tuple[typing.Any, typing.Any]] | str | bytes, **kwargs: typing.Any) -> None:
        assert len(args) < 2, 'Too many arguments.'
        value = args[0] if args else []
        if isinstance(value, str):
            super().__init__(parse_qsl(value, keep_blank_values=True), **kwargs)
        elif isinstance(value, bytes):
            super().__init__(parse_qsl(value.decode('latin-1'), keep_blank_values=True), **kwargs)
        else:
            super().__init__(*args, **kwargs)
        self._list = [(str(k), str(v)) for k, v in self._list]
        self._dict = {str(k): str(v) for k, v in self._dict.items()}

    def __str__(self) -> str:
        return urlencode(self._list)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        query_string = str(self)
        return f'{class_name}({query_string!r})'