import functools
import sys
import typing as t
from collections import abc
from itertools import chain
from markupsafe import escape  # noqa: F401
from markupsafe import Markup
from markupsafe import soft_str
from .async_utils import auto_aiter
from .async_utils import auto_await  # noqa: F401
from .exceptions import TemplateNotFound  # noqa: F401
from .exceptions import TemplateRuntimeError  # noqa: F401
from .exceptions import UndefinedError
from .nodes import EvalContext
from .utils import _PassArg
from .utils import concat
from .utils import internalcode
from .utils import missing
from .utils import Namespace  # noqa: F401
from .utils import object_type_repr
from .utils import pass_eval_context
class AsyncLoopContext(LoopContext):
    _iterator: t.AsyncIterator[t.Any]

    @staticmethod
    def _to_iterator(iterable: t.Union[t.Iterable[V], t.AsyncIterable[V]]) -> t.AsyncIterator[V]:
        return auto_aiter(iterable)

    @property
    async def length(self) -> int:
        if self._length is not None:
            return self._length
        try:
            self._length = len(self._iterable)
        except TypeError:
            iterable = [x async for x in self._iterator]
            self._iterator = self._to_iterator(iterable)
            self._length = len(iterable) + self.index + (self._after is not missing)
        return self._length

    @property
    async def revindex0(self) -> int:
        return await self.length - self.index

    @property
    async def revindex(self) -> int:
        return await self.length - self.index0

    async def _peek_next(self) -> t.Any:
        if self._after is not missing:
            return self._after
        try:
            self._after = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._after = missing
        return self._after

    @property
    async def last(self) -> bool:
        return await self._peek_next() is missing

    @property
    async def nextitem(self) -> t.Union[t.Any, 'Undefined']:
        rv = await self._peek_next()
        if rv is missing:
            return self._undefined('there is no next item')
        return rv

    def __aiter__(self) -> 'AsyncLoopContext':
        return self

    async def __anext__(self) -> t.Tuple[t.Any, 'AsyncLoopContext']:
        if self._after is not missing:
            rv = self._after
            self._after = missing
        else:
            rv = await self._iterator.__anext__()
        self.index0 += 1
        self._before = self._current
        self._current = rv
        return (rv, self)