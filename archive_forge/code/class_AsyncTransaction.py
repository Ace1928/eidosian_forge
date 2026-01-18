from __future__ import annotations
import asyncio
import contextlib
from typing import Any
from typing import AsyncIterator
from typing import Callable
from typing import Dict
from typing import Generator
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc as async_exc
from .base import asyncstartablecontext
from .base import GeneratorStartableContext
from .base import ProxyComparable
from .base import StartableContext
from .result import _ensure_sync_result
from .result import AsyncResult
from .result import AsyncScalarResult
from ... import exc
from ... import inspection
from ... import util
from ...engine import Connection
from ...engine import create_engine as _create_engine
from ...engine import create_pool_from_url as _create_pool_from_url
from ...engine import Engine
from ...engine.base import NestedTransaction
from ...engine.base import Transaction
from ...exc import ArgumentError
from ...util.concurrency import greenlet_spawn
from ...util.typing import Concatenate
from ...util.typing import ParamSpec
class AsyncTransaction(ProxyComparable[Transaction], StartableContext['AsyncTransaction']):
    """An asyncio proxy for a :class:`_engine.Transaction`."""
    __slots__ = ('connection', 'sync_transaction', 'nested')
    sync_transaction: Optional[Transaction]
    connection: AsyncConnection
    nested: bool

    def __init__(self, connection: AsyncConnection, nested: bool=False):
        self.connection = connection
        self.sync_transaction = None
        self.nested = nested

    @classmethod
    def _regenerate_proxy_for_target(cls, target: Transaction) -> AsyncTransaction:
        sync_connection = target.connection
        sync_transaction = target
        nested = isinstance(target, NestedTransaction)
        async_connection = AsyncConnection._retrieve_proxy_for_target(sync_connection)
        assert async_connection is not None
        obj = cls.__new__(cls)
        obj.connection = async_connection
        obj.sync_transaction = obj._assign_proxied(sync_transaction)
        obj.nested = nested
        return obj

    @util.ro_non_memoized_property
    def _proxied(self) -> Transaction:
        if not self.sync_transaction:
            self._raise_for_not_started()
        return self.sync_transaction

    @property
    def is_valid(self) -> bool:
        return self._proxied.is_valid

    @property
    def is_active(self) -> bool:
        return self._proxied.is_active

    async def close(self) -> None:
        """Close this :class:`.AsyncTransaction`.

        If this transaction is the base transaction in a begin/commit
        nesting, the transaction will rollback().  Otherwise, the
        method returns.

        This is used to cancel a Transaction without affecting the scope of
        an enclosing transaction.

        """
        await greenlet_spawn(self._proxied.close)

    async def rollback(self) -> None:
        """Roll back this :class:`.AsyncTransaction`."""
        await greenlet_spawn(self._proxied.rollback)

    async def commit(self) -> None:
        """Commit this :class:`.AsyncTransaction`."""
        await greenlet_spawn(self._proxied.commit)

    async def start(self, is_ctxmanager: bool=False) -> AsyncTransaction:
        """Start this :class:`_asyncio.AsyncTransaction` object's context
        outside of using a Python ``with:`` block.

        """
        self.sync_transaction = self._assign_proxied(await greenlet_spawn(self.connection._proxied.begin_nested if self.nested else self.connection._proxied.begin))
        if is_ctxmanager:
            self.sync_transaction.__enter__()
        return self

    async def __aexit__(self, type_: Any, value: Any, traceback: Any) -> None:
        await greenlet_spawn(self._proxied.__exit__, type_, value, traceback)