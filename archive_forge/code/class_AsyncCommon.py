from __future__ import annotations
import operator
from typing import Any
from typing import AsyncIterator
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from . import exc as async_exc
from ... import util
from ...engine import Result
from ...engine.result import _NO_ROW
from ...engine.result import _R
from ...engine.result import _WithKeys
from ...engine.result import FilterResult
from ...engine.result import FrozenResult
from ...engine.result import ResultMetaData
from ...engine.row import Row
from ...engine.row import RowMapping
from ...sql.base import _generative
from ...util.concurrency import greenlet_spawn
from ...util.typing import Literal
from ...util.typing import Self
class AsyncCommon(FilterResult[_R]):
    __slots__ = ()
    _real_result: Result[Any]
    _metadata: ResultMetaData

    async def close(self) -> None:
        """Close this result."""
        await greenlet_spawn(self._real_result.close)

    @property
    def closed(self) -> bool:
        """proxies the .closed attribute of the underlying result object,
        if any, else raises ``AttributeError``.

        .. versionadded:: 2.0.0b3

        """
        return self._real_result.closed