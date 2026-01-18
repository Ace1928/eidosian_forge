from __future__ import annotations
import contextlib
from enum import Enum
import itertools
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import bulk_persistence
from . import context
from . import descriptor_props
from . import exc
from . import identity
from . import loading
from . import query
from . import state as statelib
from ._typing import _O
from ._typing import insp_is_mapper
from ._typing import is_composite_class
from ._typing import is_orm_option
from ._typing import is_user_defined_option
from .base import _class_to_mapper
from .base import _none_set
from .base import _state_mapper
from .base import instance_str
from .base import LoaderCallableStatus
from .base import object_mapper
from .base import object_state
from .base import PassiveFlag
from .base import state_str
from .context import FromStatement
from .context import ORMCompileState
from .identity import IdentityMap
from .query import Query
from .state import InstanceState
from .state_changes import _StateChange
from .state_changes import _StateChangeState
from .state_changes import _StateChangeStates
from .unitofwork import UOWTransaction
from .. import engine
from .. import exc as sa_exc
from .. import sql
from .. import util
from ..engine import Connection
from ..engine import Engine
from ..engine.util import TransactionalContext
from ..event import dispatcher
from ..event import EventTarget
from ..inspection import inspect
from ..inspection import Inspectable
from ..sql import coercions
from ..sql import dml
from ..sql import roles
from ..sql import Select
from ..sql import TableClause
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import CompileState
from ..sql.schema import Table
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import IdentitySet
from ..util.typing import Literal
from ..util.typing import Protocol
class SessionTransaction(_StateChange, TransactionalContext):
    """A :class:`.Session`-level transaction.

    :class:`.SessionTransaction` is produced from the
    :meth:`_orm.Session.begin`
    and :meth:`_orm.Session.begin_nested` methods.   It's largely an internal
    object that in modern use provides a context manager for session
    transactions.

    Documentation on interacting with :class:`_orm.SessionTransaction` is
    at: :ref:`unitofwork_transaction`.


    .. versionchanged:: 1.4  The scoping and API methods to work with the
       :class:`_orm.SessionTransaction` object directly have been simplified.

    .. seealso::

        :ref:`unitofwork_transaction`

        :meth:`.Session.begin`

        :meth:`.Session.begin_nested`

        :meth:`.Session.rollback`

        :meth:`.Session.commit`

        :meth:`.Session.in_transaction`

        :meth:`.Session.in_nested_transaction`

        :meth:`.Session.get_transaction`

        :meth:`.Session.get_nested_transaction`


    """
    _rollback_exception: Optional[BaseException] = None
    _connections: Dict[Union[Engine, Connection], Tuple[Connection, Transaction, bool, bool]]
    session: Session
    _parent: Optional[SessionTransaction]
    _state: SessionTransactionState
    _new: weakref.WeakKeyDictionary[InstanceState[Any], object]
    _deleted: weakref.WeakKeyDictionary[InstanceState[Any], object]
    _dirty: weakref.WeakKeyDictionary[InstanceState[Any], object]
    _key_switches: weakref.WeakKeyDictionary[InstanceState[Any], Tuple[Any, Any]]
    origin: SessionTransactionOrigin
    'Origin of this :class:`_orm.SessionTransaction`.\n\n    Refers to a :class:`.SessionTransactionOrigin` instance which is an\n    enumeration indicating the source event that led to constructing\n    this :class:`_orm.SessionTransaction`.\n\n    .. versionadded:: 2.0\n\n    '
    nested: bool = False
    'Indicates if this is a nested, or SAVEPOINT, transaction.\n\n    When :attr:`.SessionTransaction.nested` is True, it is expected\n    that :attr:`.SessionTransaction.parent` will be present as well,\n    linking to the enclosing :class:`.SessionTransaction`.\n\n    .. seealso::\n\n        :attr:`.SessionTransaction.origin`\n\n    '

    def __init__(self, session: Session, origin: SessionTransactionOrigin, parent: Optional[SessionTransaction]=None):
        TransactionalContext._trans_ctx_check(session)
        self.session = session
        self._connections = {}
        self._parent = parent
        self.nested = nested = origin is SessionTransactionOrigin.BEGIN_NESTED
        self.origin = origin
        if session._close_state is _SessionCloseState.CLOSED:
            raise sa_exc.InvalidRequestError('This Session has been permanently closed and is unable to handle any more transaction requests.')
        if nested:
            if not parent:
                raise sa_exc.InvalidRequestError("Can't start a SAVEPOINT transaction when no existing transaction is in progress")
            self._previous_nested_transaction = session._nested_transaction
        elif origin is SessionTransactionOrigin.SUBTRANSACTION:
            assert parent is not None
        else:
            assert parent is None
        self._state = SessionTransactionState.ACTIVE
        self._take_snapshot()
        self.session._transaction = self
        self.session.dispatch.after_transaction_create(self.session, self)

    def _raise_for_prerequisite_state(self, operation_name: str, state: _StateChangeState) -> NoReturn:
        if state is SessionTransactionState.DEACTIVE:
            if self._rollback_exception:
                raise sa_exc.PendingRollbackError(f"This Session's transaction has been rolled back due to a previous exception during flush. To begin a new transaction with this Session, first issue Session.rollback(). Original exception was: {self._rollback_exception}", code='7s2a')
            else:
                raise sa_exc.InvalidRequestError("This session is in 'inactive' state, due to the SQL transaction being rolled back; no further SQL can be emitted within this transaction.")
        elif state is SessionTransactionState.CLOSED:
            raise sa_exc.ResourceClosedError('This transaction is closed')
        elif state is SessionTransactionState.PROVISIONING_CONNECTION:
            raise sa_exc.InvalidRequestError('This session is provisioning a new connection; concurrent operations are not permitted', code='isce')
        else:
            raise sa_exc.InvalidRequestError(f"This session is in '{state.name.lower()}' state; no further SQL can be emitted within this transaction.")

    @property
    def parent(self) -> Optional[SessionTransaction]:
        """The parent :class:`.SessionTransaction` of this
        :class:`.SessionTransaction`.

        If this attribute is ``None``, indicates this
        :class:`.SessionTransaction` is at the top of the stack, and
        corresponds to a real "COMMIT"/"ROLLBACK"
        block.  If non-``None``, then this is either a "subtransaction"
        (an internal marker object used by the flush process) or a
        "nested" / SAVEPOINT transaction.  If the
        :attr:`.SessionTransaction.nested` attribute is ``True``, then
        this is a SAVEPOINT, and if ``False``, indicates this a subtransaction.

        """
        return self._parent

    @property
    def is_active(self) -> bool:
        return self.session is not None and self._state is SessionTransactionState.ACTIVE

    @property
    def _is_transaction_boundary(self) -> bool:
        return self.nested or not self._parent

    @_StateChange.declare_states((SessionTransactionState.ACTIVE,), _StateChangeStates.NO_CHANGE)
    def connection(self, bindkey: Optional[Mapper[Any]], execution_options: Optional[_ExecuteOptions]=None, **kwargs: Any) -> Connection:
        bind = self.session.get_bind(bindkey, **kwargs)
        return self._connection_for_bind(bind, execution_options)

    @_StateChange.declare_states((SessionTransactionState.ACTIVE,), _StateChangeStates.NO_CHANGE)
    def _begin(self, nested: bool=False) -> SessionTransaction:
        return SessionTransaction(self.session, SessionTransactionOrigin.BEGIN_NESTED if nested else SessionTransactionOrigin.SUBTRANSACTION, self)

    def _iterate_self_and_parents(self, upto: Optional[SessionTransaction]=None) -> Iterable[SessionTransaction]:
        current = self
        result: Tuple[SessionTransaction, ...] = ()
        while current:
            result += (current,)
            if current._parent is upto:
                break
            elif current._parent is None:
                raise sa_exc.InvalidRequestError('Transaction %s is not on the active transaction list' % upto)
            else:
                current = current._parent
        return result

    def _take_snapshot(self) -> None:
        if not self._is_transaction_boundary:
            parent = self._parent
            assert parent is not None
            self._new = parent._new
            self._deleted = parent._deleted
            self._dirty = parent._dirty
            self._key_switches = parent._key_switches
            return
        is_begin = self.origin in (SessionTransactionOrigin.BEGIN, SessionTransactionOrigin.AUTOBEGIN)
        if not is_begin and (not self.session._flushing):
            self.session.flush()
        self._new = weakref.WeakKeyDictionary()
        self._deleted = weakref.WeakKeyDictionary()
        self._dirty = weakref.WeakKeyDictionary()
        self._key_switches = weakref.WeakKeyDictionary()

    def _restore_snapshot(self, dirty_only: bool=False) -> None:
        """Restore the restoration state taken before a transaction began.

        Corresponds to a rollback.

        """
        assert self._is_transaction_boundary
        to_expunge = set(self._new).union(self.session._new)
        self.session._expunge_states(to_expunge, to_transient=True)
        for s, (oldkey, newkey) in self._key_switches.items():
            self.session.identity_map.safe_discard(s)
            s.key = oldkey
            if s not in to_expunge:
                self.session.identity_map.replace(s)
        for s in set(self._deleted).union(self.session._deleted):
            self.session._update_impl(s, revert_deletion=True)
        assert not self.session._deleted
        for s in self.session.identity_map.all_states():
            if not dirty_only or s.modified or s in self._dirty:
                s._expire(s.dict, self.session.identity_map._modified)

    def _remove_snapshot(self) -> None:
        """Remove the restoration state taken before a transaction began.

        Corresponds to a commit.

        """
        assert self._is_transaction_boundary
        if not self.nested and self.session.expire_on_commit:
            for s in self.session.identity_map.all_states():
                s._expire(s.dict, self.session.identity_map._modified)
            statelib.InstanceState._detach_states(list(self._deleted), self.session)
            self._deleted.clear()
        elif self.nested:
            parent = self._parent
            assert parent is not None
            parent._new.update(self._new)
            parent._dirty.update(self._dirty)
            parent._deleted.update(self._deleted)
            parent._key_switches.update(self._key_switches)

    @_StateChange.declare_states((SessionTransactionState.ACTIVE,), _StateChangeStates.NO_CHANGE)
    def _connection_for_bind(self, bind: _SessionBind, execution_options: Optional[CoreExecuteOptionsParameter]) -> Connection:
        if bind in self._connections:
            if execution_options:
                util.warn('Connection is already established for the given bind; execution_options ignored')
            return self._connections[bind][0]
        self._state = SessionTransactionState.PROVISIONING_CONNECTION
        local_connect = False
        should_commit = True
        try:
            if self._parent:
                conn = self._parent._connection_for_bind(bind, execution_options)
                if not self.nested:
                    return conn
            elif isinstance(bind, engine.Connection):
                conn = bind
                if conn.engine in self._connections:
                    raise sa_exc.InvalidRequestError("Session already has a Connection associated for the given Connection's Engine")
            else:
                conn = bind.connect()
                local_connect = True
            try:
                if execution_options:
                    conn = conn.execution_options(**execution_options)
                transaction: Transaction
                if self.session.twophase and self._parent is None:
                    transaction = conn.begin_twophase()
                elif self.nested:
                    transaction = conn.begin_nested()
                elif conn.in_transaction():
                    join_transaction_mode = self.session.join_transaction_mode
                    if join_transaction_mode == 'conditional_savepoint':
                        if conn.in_nested_transaction():
                            join_transaction_mode = 'create_savepoint'
                        else:
                            join_transaction_mode = 'rollback_only'
                    if join_transaction_mode in ('control_fully', 'rollback_only'):
                        if conn.in_nested_transaction():
                            transaction = conn._get_required_nested_transaction()
                        else:
                            transaction = conn._get_required_transaction()
                        if join_transaction_mode == 'rollback_only':
                            should_commit = False
                    elif join_transaction_mode == 'create_savepoint':
                        transaction = conn.begin_nested()
                    else:
                        assert False, join_transaction_mode
                else:
                    transaction = conn.begin()
            except:
                if local_connect:
                    conn.close()
                raise
            else:
                bind_is_connection = isinstance(bind, engine.Connection)
                self._connections[conn] = self._connections[conn.engine] = (conn, transaction, should_commit, not bind_is_connection)
                self.session.dispatch.after_begin(self.session, self, conn)
                return conn
        finally:
            self._state = SessionTransactionState.ACTIVE

    def prepare(self) -> None:
        if self._parent is not None or not self.session.twophase:
            raise sa_exc.InvalidRequestError("'twophase' mode not enabled, or not root transaction; can't prepare.")
        self._prepare_impl()

    @_StateChange.declare_states((SessionTransactionState.ACTIVE,), SessionTransactionState.PREPARED)
    def _prepare_impl(self) -> None:
        if self._parent is None or self.nested:
            self.session.dispatch.before_commit(self.session)
        stx = self.session._transaction
        assert stx is not None
        if stx is not self:
            for subtransaction in stx._iterate_self_and_parents(upto=self):
                subtransaction.commit()
        if not self.session._flushing:
            for _flush_guard in range(100):
                if self.session._is_clean():
                    break
                self.session.flush()
            else:
                raise exc.FlushError('Over 100 subsequent flushes have occurred within session.commit() - is an after_flush() hook creating new objects?')
        if self._parent is None and self.session.twophase:
            try:
                for t in set(self._connections.values()):
                    cast('TwoPhaseTransaction', t[1]).prepare()
            except:
                with util.safe_reraise():
                    self.rollback()
        self._state = SessionTransactionState.PREPARED

    @_StateChange.declare_states((SessionTransactionState.ACTIVE, SessionTransactionState.PREPARED), SessionTransactionState.CLOSED)
    def commit(self, _to_root: bool=False) -> None:
        if self._state is not SessionTransactionState.PREPARED:
            with self._expect_state(SessionTransactionState.PREPARED):
                self._prepare_impl()
        if self._parent is None or self.nested:
            for conn, trans, should_commit, autoclose in set(self._connections.values()):
                if should_commit:
                    trans.commit()
            self._state = SessionTransactionState.COMMITTED
            self.session.dispatch.after_commit(self.session)
            self._remove_snapshot()
        with self._expect_state(SessionTransactionState.CLOSED):
            self.close()
        if _to_root and self._parent:
            self._parent.commit(_to_root=True)

    @_StateChange.declare_states((SessionTransactionState.ACTIVE, SessionTransactionState.DEACTIVE, SessionTransactionState.PREPARED), SessionTransactionState.CLOSED)
    def rollback(self, _capture_exception: bool=False, _to_root: bool=False) -> None:
        stx = self.session._transaction
        assert stx is not None
        if stx is not self:
            for subtransaction in stx._iterate_self_and_parents(upto=self):
                subtransaction.close()
        boundary = self
        rollback_err = None
        if self._state in (SessionTransactionState.ACTIVE, SessionTransactionState.PREPARED):
            for transaction in self._iterate_self_and_parents():
                if transaction._parent is None or transaction.nested:
                    try:
                        for t in set(transaction._connections.values()):
                            t[1].rollback()
                        transaction._state = SessionTransactionState.DEACTIVE
                        self.session.dispatch.after_rollback(self.session)
                    except:
                        rollback_err = sys.exc_info()
                    finally:
                        transaction._state = SessionTransactionState.DEACTIVE
                        transaction._restore_snapshot(dirty_only=transaction.nested)
                    boundary = transaction
                    break
                else:
                    transaction._state = SessionTransactionState.DEACTIVE
        sess = self.session
        if not rollback_err and (not sess._is_clean()):
            util.warn("Session's state has been changed on a non-active transaction - this state will be discarded.")
            boundary._restore_snapshot(dirty_only=boundary.nested)
        with self._expect_state(SessionTransactionState.CLOSED):
            self.close()
        if self._parent and _capture_exception:
            self._parent._rollback_exception = sys.exc_info()[1]
        if rollback_err and rollback_err[1]:
            raise rollback_err[1].with_traceback(rollback_err[2])
        sess.dispatch.after_soft_rollback(sess, self)
        if _to_root and self._parent:
            self._parent.rollback(_to_root=True)

    @_StateChange.declare_states(_StateChangeStates.ANY, SessionTransactionState.CLOSED)
    def close(self, invalidate: bool=False) -> None:
        if self.nested:
            self.session._nested_transaction = self._previous_nested_transaction
        self.session._transaction = self._parent
        for connection, transaction, should_commit, autoclose in set(self._connections.values()):
            if invalidate and self._parent is None:
                connection.invalidate()
            if should_commit and transaction.is_active:
                transaction.close()
            if autoclose and self._parent is None:
                connection.close()
        self._state = SessionTransactionState.CLOSED
        sess = self.session
        sess.dispatch.after_transaction_end(sess, self)

    def _get_subject(self) -> Session:
        return self.session

    def _transaction_is_active(self) -> bool:
        return self._state is SessionTransactionState.ACTIVE

    def _transaction_is_closed(self) -> bool:
        return self._state is SessionTransactionState.CLOSED

    def _rollback_can_be_called(self) -> bool:
        return self._state not in (COMMITTED, CLOSED)