from __future__ import annotations
import contextlib
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from .interfaces import BindTyping
from .interfaces import ConnectionEventsTarget
from .interfaces import DBAPICursor
from .interfaces import ExceptionContext
from .interfaces import ExecuteStyle
from .interfaces import ExecutionContext
from .interfaces import IsolationLevel
from .util import _distill_params_20
from .util import _distill_raw_params
from .util import TransactionalContext
from .. import exc
from .. import inspection
from .. import log
from .. import util
from ..sql import compiler
from ..sql import util as sql_util
class TwoPhaseTransaction(RootTransaction):
    """Represent a two-phase transaction.

    A new :class:`.TwoPhaseTransaction` object may be procured
    using the :meth:`_engine.Connection.begin_twophase` method.

    The interface is the same as that of :class:`.Transaction`
    with the addition of the :meth:`prepare` method.

    """
    __slots__ = ('xid', '_is_prepared')
    xid: Any

    def __init__(self, connection: Connection, xid: Any):
        self._is_prepared = False
        self.xid = xid
        super().__init__(connection)

    def prepare(self) -> None:
        """Prepare this :class:`.TwoPhaseTransaction`.

        After a PREPARE, the transaction can be committed.

        """
        if not self.is_active:
            raise exc.InvalidRequestError('This transaction is inactive')
        self.connection._prepare_twophase_impl(self.xid)
        self._is_prepared = True

    def _connection_begin_impl(self) -> None:
        self.connection._begin_twophase_impl(self)

    def _connection_rollback_impl(self) -> None:
        self.connection._rollback_twophase_impl(self.xid, self._is_prepared)

    def _connection_commit_impl(self) -> None:
        self.connection._commit_twophase_impl(self.xid, self._is_prepared)