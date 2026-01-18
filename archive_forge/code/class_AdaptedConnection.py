from __future__ import annotations
from enum import Enum
from types import ModuleType
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..event import EventTarget
from ..pool import Pool
from ..pool import PoolProxiedConnection
from ..sql.compiler import Compiled as Compiled
from ..sql.compiler import Compiled  # noqa
from ..sql.compiler import TypeCompiler as TypeCompiler
from ..sql.compiler import TypeCompiler  # noqa
from ..util import immutabledict
from ..util.concurrency import await_only
from ..util.typing import Literal
from ..util.typing import NotRequired
from ..util.typing import Protocol
from ..util.typing import TypedDict
class AdaptedConnection:
    """Interface of an adapted connection object to support the DBAPI protocol.

    Used by asyncio dialects to provide a sync-style pep-249 facade on top
    of the asyncio connection/cursor API provided by the driver.

    .. versionadded:: 1.4.24

    """
    __slots__ = ('_connection',)
    _connection: Any

    @property
    def driver_connection(self) -> Any:
        """The connection object as returned by the driver after a connect."""
        return self._connection

    def run_async(self, fn: Callable[[Any], Awaitable[_T]]) -> _T:
        """Run the awaitable returned by the given function, which is passed
        the raw asyncio driver connection.

        This is used to invoke awaitable-only methods on the driver connection
        within the context of a "synchronous" method, like a connection
        pool event handler.

        E.g.::

            engine = create_async_engine(...)

            @event.listens_for(engine.sync_engine, "connect")
            def register_custom_types(dbapi_connection, ...):
                dbapi_connection.run_async(
                    lambda connection: connection.set_type_codec(
                        'MyCustomType', encoder, decoder, ...
                    )
                )

        .. versionadded:: 1.4.30

        .. seealso::

            :ref:`asyncio_events_run_async`

        """
        return await_only(fn(self._connection))

    def __repr__(self) -> str:
        return '<AdaptedConnection %s>' % self._connection