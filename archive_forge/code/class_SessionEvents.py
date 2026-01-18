from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import instrumentation
from . import interfaces
from . import mapperlib
from .attributes import QueryableAttribute
from .base import _mapper_or_none
from .base import NO_KEY
from .instrumentation import ClassManager
from .instrumentation import InstrumentationFactory
from .query import BulkDelete
from .query import BulkUpdate
from .query import Query
from .scoping import scoped_session
from .session import Session
from .session import sessionmaker
from .. import event
from .. import exc
from .. import util
from ..event import EventTarget
from ..event.registry import _ET
from ..util.compat import inspect_getfullargspec
class SessionEvents(event.Events[Session]):
    """Define events specific to :class:`.Session` lifecycle.

    e.g.::

        from sqlalchemy import event
        from sqlalchemy.orm import sessionmaker

        def my_before_commit(session):
            print("before commit!")

        Session = sessionmaker()

        event.listen(Session, "before_commit", my_before_commit)

    The :func:`~.event.listen` function will accept
    :class:`.Session` objects as well as the return result
    of :class:`~.sessionmaker()` and :class:`~.scoped_session()`.

    Additionally, it accepts the :class:`.Session` class which
    will apply listeners to all :class:`.Session` instances
    globally.

    :param raw=False: When True, the "target" argument passed
       to applicable event listener functions that work on individual
       objects will be the instance's :class:`.InstanceState` management
       object, rather than the mapped instance itself.

       .. versionadded:: 1.3.14

    :param restore_load_context=False: Applies to the
       :meth:`.SessionEvents.loaded_as_persistent` event.  Restores the loader
       context of the object when the event hook is complete, so that ongoing
       eager load operations continue to target the object appropriately.  A
       warning is emitted if the object is moved to a new loader context from
       within this event if this flag is not set.

       .. versionadded:: 1.3.14

    """
    _target_class_doc = 'SomeSessionClassOrObject'
    _dispatch_target = Session

    def _lifecycle_event(fn: Callable[[SessionEvents, Session, Any], None]) -> Callable[[SessionEvents, Session, Any], None]:
        _sessionevents_lifecycle_event_names.add(fn.__name__)
        return fn

    @classmethod
    def _accept_with(cls, target: Any, identifier: str) -> Union[Session, type]:
        if isinstance(target, scoped_session):
            target = target.session_factory
            if not isinstance(target, sessionmaker) and (not isinstance(target, type) or not issubclass(target, Session)):
                raise exc.ArgumentError('Session event listen on a scoped_session requires that its creation callable is associated with the Session class.')
        if isinstance(target, sessionmaker):
            return target.class_
        elif isinstance(target, type):
            if issubclass(target, scoped_session):
                return Session
            elif issubclass(target, Session):
                return target
        elif isinstance(target, Session):
            return target
        elif hasattr(target, '_no_async_engine_events'):
            target._no_async_engine_events()
        else:
            return event.Events._accept_with(target, identifier)

    @classmethod
    def _listen(cls, event_key: Any, *, raw: bool=False, restore_load_context: bool=False, **kw: Any) -> None:
        is_instance_event = event_key.identifier in _sessionevents_lifecycle_event_names
        if is_instance_event:
            if not raw or restore_load_context:
                fn = event_key._listen_fn

                def wrap(session: Session, state: InstanceState[_O], *arg: Any, **kw: Any) -> Optional[Any]:
                    if not raw:
                        target = state.obj()
                        if target is None:
                            return None
                    else:
                        target = state
                    if restore_load_context:
                        runid = state.runid
                    try:
                        return fn(session, target, *arg, **kw)
                    finally:
                        if restore_load_context:
                            state.runid = runid
                event_key = event_key.with_wrapper(wrap)
        event_key.base_listen(**kw)

    def do_orm_execute(self, orm_execute_state: ORMExecuteState) -> None:
        """Intercept statement executions that occur on behalf of an
        ORM :class:`.Session` object.

        This event is invoked for all top-level SQL statements invoked from the
        :meth:`_orm.Session.execute` method, as well as related methods such as
        :meth:`_orm.Session.scalars` and :meth:`_orm.Session.scalar`. As of
        SQLAlchemy 1.4, all ORM queries that run through the
        :meth:`_orm.Session.execute` method as well as related methods
        :meth:`_orm.Session.scalars`, :meth:`_orm.Session.scalar` etc.
        will participate in this event.
        This event hook does **not** apply to the queries that are
        emitted internally within the ORM flush process, i.e. the
        process described at :ref:`session_flushing`.

        .. note::  The :meth:`_orm.SessionEvents.do_orm_execute` event hook
           is triggered **for ORM statement executions only**, meaning those
           invoked via the :meth:`_orm.Session.execute` and similar methods on
           the :class:`_orm.Session` object. It does **not** trigger for
           statements that are invoked by SQLAlchemy Core only, i.e. statements
           invoked directly using :meth:`_engine.Connection.execute` or
           otherwise originating from an :class:`_engine.Engine` object without
           any :class:`_orm.Session` involved. To intercept **all** SQL
           executions regardless of whether the Core or ORM APIs are in use,
           see the event hooks at :class:`.ConnectionEvents`, such as
           :meth:`.ConnectionEvents.before_execute` and
           :meth:`.ConnectionEvents.before_cursor_execute`.

           Also, this event hook does **not** apply to queries that are
           emitted internally within the ORM flush process,
           i.e. the process described at :ref:`session_flushing`; to
           intercept steps within the flush process, see the event
           hooks described at :ref:`session_persistence_events` as
           well as :ref:`session_persistence_mapper`.

        This event is a ``do_`` event, meaning it has the capability to replace
        the operation that the :meth:`_orm.Session.execute` method normally
        performs.  The intended use for this includes sharding and
        result-caching schemes which may seek to invoke the same statement
        across  multiple database connections, returning a result that is
        merged from each of them, or which don't invoke the statement at all,
        instead returning data from a cache.

        The hook intends to replace the use of the
        ``Query._execute_and_instances`` method that could be subclassed prior
        to SQLAlchemy 1.4.

        :param orm_execute_state: an instance of :class:`.ORMExecuteState`
         which contains all information about the current execution, as well
         as helper functions used to derive other commonly required
         information.   See that object for details.

        .. seealso::

            :ref:`session_execute_events` - top level documentation on how
            to use :meth:`_orm.SessionEvents.do_orm_execute`

            :class:`.ORMExecuteState` - the object passed to the
            :meth:`_orm.SessionEvents.do_orm_execute` event which contains
            all information about the statement to be invoked.  It also
            provides an interface to extend the current statement, options,
            and parameters as well as an option that allows programmatic
            invocation of the statement at any point.

            :ref:`examples_session_orm_events` - includes examples of using
            :meth:`_orm.SessionEvents.do_orm_execute`

            :ref:`examples_caching` - an example of how to integrate
            Dogpile caching with the ORM :class:`_orm.Session` making use
            of the :meth:`_orm.SessionEvents.do_orm_execute` event hook.

            :ref:`examples_sharding` - the Horizontal Sharding example /
            extension relies upon the
            :meth:`_orm.SessionEvents.do_orm_execute` event hook to invoke a
            SQL statement on multiple backends and return a merged result.


        .. versionadded:: 1.4

        """

    def after_transaction_create(self, session: Session, transaction: SessionTransaction) -> None:
        """Execute when a new :class:`.SessionTransaction` is created.

        This event differs from :meth:`~.SessionEvents.after_begin`
        in that it occurs for each :class:`.SessionTransaction`
        overall, as opposed to when transactions are begun
        on individual database connections.  It is also invoked
        for nested transactions and subtransactions, and is always
        matched by a corresponding
        :meth:`~.SessionEvents.after_transaction_end` event
        (assuming normal operation of the :class:`.Session`).

        :param session: the target :class:`.Session`.
        :param transaction: the target :class:`.SessionTransaction`.

         To detect if this is the outermost
         :class:`.SessionTransaction`, as opposed to a "subtransaction" or a
         SAVEPOINT, test that the :attr:`.SessionTransaction.parent` attribute
         is ``None``::

                @event.listens_for(session, "after_transaction_create")
                def after_transaction_create(session, transaction):
                    if transaction.parent is None:
                        # work with top-level transaction

         To detect if the :class:`.SessionTransaction` is a SAVEPOINT, use the
         :attr:`.SessionTransaction.nested` attribute::

                @event.listens_for(session, "after_transaction_create")
                def after_transaction_create(session, transaction):
                    if transaction.nested:
                        # work with SAVEPOINT transaction


        .. seealso::

            :class:`.SessionTransaction`

            :meth:`~.SessionEvents.after_transaction_end`

        """

    def after_transaction_end(self, session: Session, transaction: SessionTransaction) -> None:
        """Execute when the span of a :class:`.SessionTransaction` ends.

        This event differs from :meth:`~.SessionEvents.after_commit`
        in that it corresponds to all :class:`.SessionTransaction`
        objects in use, including those for nested transactions
        and subtransactions, and is always matched by a corresponding
        :meth:`~.SessionEvents.after_transaction_create` event.

        :param session: the target :class:`.Session`.
        :param transaction: the target :class:`.SessionTransaction`.

         To detect if this is the outermost
         :class:`.SessionTransaction`, as opposed to a "subtransaction" or a
         SAVEPOINT, test that the :attr:`.SessionTransaction.parent` attribute
         is ``None``::

                @event.listens_for(session, "after_transaction_create")
                def after_transaction_end(session, transaction):
                    if transaction.parent is None:
                        # work with top-level transaction

         To detect if the :class:`.SessionTransaction` is a SAVEPOINT, use the
         :attr:`.SessionTransaction.nested` attribute::

                @event.listens_for(session, "after_transaction_create")
                def after_transaction_end(session, transaction):
                    if transaction.nested:
                        # work with SAVEPOINT transaction


        .. seealso::

            :class:`.SessionTransaction`

            :meth:`~.SessionEvents.after_transaction_create`

        """

    def before_commit(self, session: Session) -> None:
        """Execute before commit is called.

        .. note::

            The :meth:`~.SessionEvents.before_commit` hook is *not* per-flush,
            that is, the :class:`.Session` can emit SQL to the database
            many times within the scope of a transaction.
            For interception of these events, use the
            :meth:`~.SessionEvents.before_flush`,
            :meth:`~.SessionEvents.after_flush`, or
            :meth:`~.SessionEvents.after_flush_postexec`
            events.

        :param session: The target :class:`.Session`.

        .. seealso::

            :meth:`~.SessionEvents.after_commit`

            :meth:`~.SessionEvents.after_begin`

            :meth:`~.SessionEvents.after_transaction_create`

            :meth:`~.SessionEvents.after_transaction_end`

        """

    def after_commit(self, session: Session) -> None:
        """Execute after a commit has occurred.

        .. note::

            The :meth:`~.SessionEvents.after_commit` hook is *not* per-flush,
            that is, the :class:`.Session` can emit SQL to the database
            many times within the scope of a transaction.
            For interception of these events, use the
            :meth:`~.SessionEvents.before_flush`,
            :meth:`~.SessionEvents.after_flush`, or
            :meth:`~.SessionEvents.after_flush_postexec`
            events.

        .. note::

            The :class:`.Session` is not in an active transaction
            when the :meth:`~.SessionEvents.after_commit` event is invoked,
            and therefore can not emit SQL.  To emit SQL corresponding to
            every transaction, use the :meth:`~.SessionEvents.before_commit`
            event.

        :param session: The target :class:`.Session`.

        .. seealso::

            :meth:`~.SessionEvents.before_commit`

            :meth:`~.SessionEvents.after_begin`

            :meth:`~.SessionEvents.after_transaction_create`

            :meth:`~.SessionEvents.after_transaction_end`

        """

    def after_rollback(self, session: Session) -> None:
        """Execute after a real DBAPI rollback has occurred.

        Note that this event only fires when the *actual* rollback against
        the database occurs - it does *not* fire each time the
        :meth:`.Session.rollback` method is called, if the underlying
        DBAPI transaction has already been rolled back.  In many
        cases, the :class:`.Session` will not be in
        an "active" state during this event, as the current
        transaction is not valid.   To acquire a :class:`.Session`
        which is active after the outermost rollback has proceeded,
        use the :meth:`.SessionEvents.after_soft_rollback` event, checking the
        :attr:`.Session.is_active` flag.

        :param session: The target :class:`.Session`.

        """

    def after_soft_rollback(self, session: Session, previous_transaction: SessionTransaction) -> None:
        """Execute after any rollback has occurred, including "soft"
        rollbacks that don't actually emit at the DBAPI level.

        This corresponds to both nested and outer rollbacks, i.e.
        the innermost rollback that calls the DBAPI's
        rollback() method, as well as the enclosing rollback
        calls that only pop themselves from the transaction stack.

        The given :class:`.Session` can be used to invoke SQL and
        :meth:`.Session.query` operations after an outermost rollback
        by first checking the :attr:`.Session.is_active` flag::

            @event.listens_for(Session, "after_soft_rollback")
            def do_something(session, previous_transaction):
                if session.is_active:
                    session.execute(text("select * from some_table"))

        :param session: The target :class:`.Session`.
        :param previous_transaction: The :class:`.SessionTransaction`
         transactional marker object which was just closed.   The current
         :class:`.SessionTransaction` for the given :class:`.Session` is
         available via the :attr:`.Session.transaction` attribute.

        """

    def before_flush(self, session: Session, flush_context: UOWTransaction, instances: Optional[Sequence[_O]]) -> None:
        """Execute before flush process has started.

        :param session: The target :class:`.Session`.
        :param flush_context: Internal :class:`.UOWTransaction` object
         which handles the details of the flush.
        :param instances: Usually ``None``, this is the collection of
         objects which can be passed to the :meth:`.Session.flush` method
         (note this usage is deprecated).

        .. seealso::

            :meth:`~.SessionEvents.after_flush`

            :meth:`~.SessionEvents.after_flush_postexec`

            :ref:`session_persistence_events`

        """

    def after_flush(self, session: Session, flush_context: UOWTransaction) -> None:
        """Execute after flush has completed, but before commit has been
        called.

        Note that the session's state is still in pre-flush, i.e. 'new',
        'dirty', and 'deleted' lists still show pre-flush state as well
        as the history settings on instance attributes.

        .. warning:: This event runs after the :class:`.Session` has emitted
           SQL to modify the database, but **before** it has altered its
           internal state to reflect those changes, including that newly
           inserted objects are placed into the identity map.  ORM operations
           emitted within this event such as loads of related items
           may produce new identity map entries that will immediately
           be replaced, sometimes causing confusing results.  SQLAlchemy will
           emit a warning for this condition as of version 1.3.9.

        :param session: The target :class:`.Session`.
        :param flush_context: Internal :class:`.UOWTransaction` object
         which handles the details of the flush.

        .. seealso::

            :meth:`~.SessionEvents.before_flush`

            :meth:`~.SessionEvents.after_flush_postexec`

            :ref:`session_persistence_events`

        """

    def after_flush_postexec(self, session: Session, flush_context: UOWTransaction) -> None:
        """Execute after flush has completed, and after the post-exec
        state occurs.

        This will be when the 'new', 'dirty', and 'deleted' lists are in
        their final state.  An actual commit() may or may not have
        occurred, depending on whether or not the flush started its own
        transaction or participated in a larger transaction.

        :param session: The target :class:`.Session`.
        :param flush_context: Internal :class:`.UOWTransaction` object
         which handles the details of the flush.


        .. seealso::

            :meth:`~.SessionEvents.before_flush`

            :meth:`~.SessionEvents.after_flush`

            :ref:`session_persistence_events`

        """

    def after_begin(self, session: Session, transaction: SessionTransaction, connection: Connection) -> None:
        """Execute after a transaction is begun on a connection.

        .. note:: This event is called within the process of the
          :class:`_orm.Session` modifying its own internal state.
          To invoke SQL operations within this hook, use the
          :class:`_engine.Connection` provided to the event;
          do not run SQL operations using the :class:`_orm.Session`
          directly.

        :param session: The target :class:`.Session`.
        :param transaction: The :class:`.SessionTransaction`.
        :param connection: The :class:`_engine.Connection` object
         which will be used for SQL statements.

        .. seealso::

            :meth:`~.SessionEvents.before_commit`

            :meth:`~.SessionEvents.after_commit`

            :meth:`~.SessionEvents.after_transaction_create`

            :meth:`~.SessionEvents.after_transaction_end`

        """

    @_lifecycle_event
    def before_attach(self, session: Session, instance: _O) -> None:
        """Execute before an instance is attached to a session.

        This is called before an add, delete or merge causes
        the object to be part of the session.

        .. seealso::

            :meth:`~.SessionEvents.after_attach`

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def after_attach(self, session: Session, instance: _O) -> None:
        """Execute after an instance is attached to a session.

        This is called after an add, delete or merge.

        .. note::

           As of 0.8, this event fires off *after* the item
           has been fully associated with the session, which is
           different than previous releases.  For event
           handlers that require the object not yet
           be part of session state (such as handlers which
           may autoflush while the target object is not
           yet complete) consider the
           new :meth:`.before_attach` event.

        .. seealso::

            :meth:`~.SessionEvents.before_attach`

            :ref:`session_lifecycle_events`

        """

    @event._legacy_signature('0.9', ['session', 'query', 'query_context', 'result'], lambda update_context: (update_context.session, update_context.query, None, update_context.result))
    def after_bulk_update(self, update_context: _O) -> None:
        """Event for after the legacy :meth:`_orm.Query.update` method
        has been called.

        .. legacy:: The :meth:`_orm.SessionEvents.after_bulk_update` method
           is a legacy event hook as of SQLAlchemy 2.0.   The event
           **does not participate** in :term:`2.0 style` invocations
           using :func:`_dml.update` documented at
           :ref:`orm_queryguide_update_delete_where`.  For 2.0 style use,
           the :meth:`_orm.SessionEvents.do_orm_execute` hook will intercept
           these calls.

        :param update_context: an "update context" object which contains
         details about the update, including these attributes:

            * ``session`` - the :class:`.Session` involved
            * ``query`` -the :class:`_query.Query`
              object that this update operation
              was called upon.
            * ``values`` The "values" dictionary that was passed to
              :meth:`_query.Query.update`.
            * ``result`` the :class:`_engine.CursorResult`
              returned as a result of the
              bulk UPDATE operation.

        .. versionchanged:: 1.4 the update_context no longer has a
           ``QueryContext`` object associated with it.

        .. seealso::

            :meth:`.QueryEvents.before_compile_update`

            :meth:`.SessionEvents.after_bulk_delete`

        """

    @event._legacy_signature('0.9', ['session', 'query', 'query_context', 'result'], lambda delete_context: (delete_context.session, delete_context.query, None, delete_context.result))
    def after_bulk_delete(self, delete_context: _O) -> None:
        """Event for after the legacy :meth:`_orm.Query.delete` method
        has been called.

        .. legacy:: The :meth:`_orm.SessionEvents.after_bulk_delete` method
           is a legacy event hook as of SQLAlchemy 2.0.   The event
           **does not participate** in :term:`2.0 style` invocations
           using :func:`_dml.delete` documented at
           :ref:`orm_queryguide_update_delete_where`.  For 2.0 style use,
           the :meth:`_orm.SessionEvents.do_orm_execute` hook will intercept
           these calls.

        :param delete_context: a "delete context" object which contains
         details about the update, including these attributes:

            * ``session`` - the :class:`.Session` involved
            * ``query`` -the :class:`_query.Query`
              object that this update operation
              was called upon.
            * ``result`` the :class:`_engine.CursorResult`
              returned as a result of the
              bulk DELETE operation.

        .. versionchanged:: 1.4 the update_context no longer has a
           ``QueryContext`` object associated with it.

        .. seealso::

            :meth:`.QueryEvents.before_compile_delete`

            :meth:`.SessionEvents.after_bulk_update`

        """

    @_lifecycle_event
    def transient_to_pending(self, session: Session, instance: _O) -> None:
        """Intercept the "transient to pending" transition for a specific
        object.

        This event is a specialization of the
        :meth:`.SessionEvents.after_attach` event which is only invoked
        for this specific transition.  It is invoked typically during the
        :meth:`.Session.add` call.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def pending_to_transient(self, session: Session, instance: _O) -> None:
        """Intercept the "pending to transient" transition for a specific
        object.

        This less common transition occurs when an pending object that has
        not been flushed is evicted from the session; this can occur
        when the :meth:`.Session.rollback` method rolls back the transaction,
        or when the :meth:`.Session.expunge` method is used.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def persistent_to_transient(self, session: Session, instance: _O) -> None:
        """Intercept the "persistent to transient" transition for a specific
        object.

        This less common transition occurs when an pending object that has
        has been flushed is evicted from the session; this can occur
        when the :meth:`.Session.rollback` method rolls back the transaction.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def pending_to_persistent(self, session: Session, instance: _O) -> None:
        """Intercept the "pending to persistent"" transition for a specific
        object.

        This event is invoked within the flush process, and is
        similar to scanning the :attr:`.Session.new` collection within
        the :meth:`.SessionEvents.after_flush` event.  However, in this
        case the object has already been moved to the persistent state
        when the event is called.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def detached_to_persistent(self, session: Session, instance: _O) -> None:
        """Intercept the "detached to persistent" transition for a specific
        object.

        This event is a specialization of the
        :meth:`.SessionEvents.after_attach` event which is only invoked
        for this specific transition.  It is invoked typically during the
        :meth:`.Session.add` call, as well as during the
        :meth:`.Session.delete` call if the object was not previously
        associated with the
        :class:`.Session` (note that an object marked as "deleted" remains
        in the "persistent" state until the flush proceeds).

        .. note::

            If the object becomes persistent as part of a call to
            :meth:`.Session.delete`, the object is **not** yet marked as
            deleted when this event is called.  To detect deleted objects,
            check the ``deleted`` flag sent to the
            :meth:`.SessionEvents.persistent_to_detached` to event after the
            flush proceeds, or check the :attr:`.Session.deleted` collection
            within the :meth:`.SessionEvents.before_flush` event if deleted
            objects need to be intercepted before the flush.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def loaded_as_persistent(self, session: Session, instance: _O) -> None:
        """Intercept the "loaded as persistent" transition for a specific
        object.

        This event is invoked within the ORM loading process, and is invoked
        very similarly to the :meth:`.InstanceEvents.load` event.  However,
        the event here is linkable to a :class:`.Session` class or instance,
        rather than to a mapper or class hierarchy, and integrates
        with the other session lifecycle events smoothly.  The object
        is guaranteed to be present in the session's identity map when
        this event is called.

        .. note:: This event is invoked within the loader process before
           eager loaders may have been completed, and the object's state may
           not be complete.  Additionally, invoking row-level refresh
           operations on the object will place the object into a new loader
           context, interfering with the existing load context.   See the note
           on :meth:`.InstanceEvents.load` for background on making use of the
           :paramref:`.SessionEvents.restore_load_context` parameter, which
           works in the same manner as that of
           :paramref:`.InstanceEvents.restore_load_context`, in  order to
           resolve this scenario.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def persistent_to_deleted(self, session: Session, instance: _O) -> None:
        """Intercept the "persistent to deleted" transition for a specific
        object.

        This event is invoked when a persistent object's identity
        is deleted from the database within a flush, however the object
        still remains associated with the :class:`.Session` until the
        transaction completes.

        If the transaction is rolled back, the object moves again
        to the persistent state, and the
        :meth:`.SessionEvents.deleted_to_persistent` event is called.
        If the transaction is committed, the object becomes detached,
        which will emit the :meth:`.SessionEvents.deleted_to_detached`
        event.

        Note that while the :meth:`.Session.delete` method is the primary
        public interface to mark an object as deleted, many objects
        get deleted due to cascade rules, which are not always determined
        until flush time.  Therefore, there's no way to catch
        every object that will be deleted until the flush has proceeded.
        the :meth:`.SessionEvents.persistent_to_deleted` event is therefore
        invoked at the end of a flush.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def deleted_to_persistent(self, session: Session, instance: _O) -> None:
        """Intercept the "deleted to persistent" transition for a specific
        object.

        This transition occurs only when an object that's been deleted
        successfully in a flush is restored due to a call to
        :meth:`.Session.rollback`.   The event is not called under
        any other circumstances.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def deleted_to_detached(self, session: Session, instance: _O) -> None:
        """Intercept the "deleted to detached" transition for a specific
        object.

        This event is invoked when a deleted object is evicted
        from the session.   The typical case when this occurs is when
        the transaction for a :class:`.Session` in which the object
        was deleted is committed; the object moves from the deleted
        state to the detached state.

        It is also invoked for objects that were deleted in a flush
        when the :meth:`.Session.expunge_all` or :meth:`.Session.close`
        events are called, as well as if the object is individually
        expunged from its deleted state via :meth:`.Session.expunge`.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def persistent_to_detached(self, session: Session, instance: _O) -> None:
        """Intercept the "persistent to detached" transition for a specific
        object.

        This event is invoked when a persistent object is evicted
        from the session.  There are many conditions that cause this
        to happen, including:

        * using a method such as :meth:`.Session.expunge`
          or :meth:`.Session.close`

        * Calling the :meth:`.Session.rollback` method, when the object
          was part of an INSERT statement for that session's transaction


        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        :param deleted: boolean.  If True, indicates this object moved
         to the detached state because it was marked as deleted and flushed.


        .. seealso::

            :ref:`session_lifecycle_events`

        """