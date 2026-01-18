from __future__ import annotations
from collections import deque
import dataclasses
from enum import Enum
import threading
import time
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from .. import event
from .. import exc
from .. import log
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
class PoolProxiedConnection(ManagesConnection):
    """A connection-like adapter for a :pep:`249` DBAPI connection, which
    includes additional methods specific to the :class:`.Pool` implementation.

    :class:`.PoolProxiedConnection` is the public-facing interface for the
    internal :class:`._ConnectionFairy` implementation object; users familiar
    with :class:`._ConnectionFairy` can consider this object to be equivalent.

    .. versionadded:: 2.0  :class:`.PoolProxiedConnection` provides the public-
       facing interface for the :class:`._ConnectionFairy` internal class.

    """
    __slots__ = ()
    if typing.TYPE_CHECKING:

        def commit(self) -> None:
            ...

        def cursor(self) -> DBAPICursor:
            ...

        def rollback(self) -> None:
            ...

    @property
    def is_valid(self) -> bool:
        """Return True if this :class:`.PoolProxiedConnection` still refers
        to an active DBAPI connection."""
        raise NotImplementedError()

    @property
    def is_detached(self) -> bool:
        """Return True if this :class:`.PoolProxiedConnection` is detached
        from its pool."""
        raise NotImplementedError()

    def detach(self) -> None:
        """Separate this connection from its Pool.

        This means that the connection will no longer be returned to the
        pool when closed, and will instead be literally closed.  The
        associated :class:`.ConnectionPoolEntry` is de-associated from this
        DBAPI connection.

        Note that any overall connection limiting constraints imposed by a
        Pool implementation may be violated after a detach, as the detached
        connection is removed from the pool's knowledge and control.

        """
        raise NotImplementedError()

    def close(self) -> None:
        """Release this connection back to the pool.

        The :meth:`.PoolProxiedConnection.close` method shadows the
        :pep:`249` ``.close()`` method, altering its behavior to instead
        :term:`release` the proxied connection back to the connection pool.

        Upon release to the pool, whether the connection stays "opened" and
        pooled in the Python process, versus actually closed out and removed
        from the Python process, is based on the pool implementation in use and
        its configuration and current state.

        """
        raise NotImplementedError()