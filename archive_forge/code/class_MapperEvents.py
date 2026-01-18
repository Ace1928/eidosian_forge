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
class MapperEvents(event.Events[mapperlib.Mapper[Any]]):
    """Define events specific to mappings.

    e.g.::

        from sqlalchemy import event

        def my_before_insert_listener(mapper, connection, target):
            # execute a stored procedure upon INSERT,
            # apply the value to the row to be inserted
            target.calculated_value = connection.execute(
                text("select my_special_function(%d)" % target.special_number)
            ).scalar()

        # associate the listener function with SomeClass,
        # to execute during the "before_insert" hook
        event.listen(
            SomeClass, 'before_insert', my_before_insert_listener)

    Available targets include:

    * mapped classes
    * unmapped superclasses of mapped or to-be-mapped classes
      (using the ``propagate=True`` flag)
    * :class:`_orm.Mapper` objects
    * the :class:`_orm.Mapper` class itself indicates listening for all
      mappers.

    Mapper events provide hooks into critical sections of the
    mapper, including those related to object instrumentation,
    object loading, and object persistence. In particular, the
    persistence methods :meth:`~.MapperEvents.before_insert`,
    and :meth:`~.MapperEvents.before_update` are popular
    places to augment the state being persisted - however, these
    methods operate with several significant restrictions. The
    user is encouraged to evaluate the
    :meth:`.SessionEvents.before_flush` and
    :meth:`.SessionEvents.after_flush` methods as more
    flexible and user-friendly hooks in which to apply
    additional database state during a flush.

    When using :class:`.MapperEvents`, several modifiers are
    available to the :func:`.event.listen` function.

    :param propagate=False: When True, the event listener should
       be applied to all inheriting mappers and/or the mappers of
       inheriting classes, as well as any
       mapper which is the target of this listener.
    :param raw=False: When True, the "target" argument passed
       to applicable event listener functions will be the
       instance's :class:`.InstanceState` management
       object, rather than the mapped instance itself.
    :param retval=False: when True, the user-defined event function
       must have a return value, the purpose of which is either to
       control subsequent event propagation, or to otherwise alter
       the operation in progress by the mapper.   Possible return
       values are:

       * ``sqlalchemy.orm.interfaces.EXT_CONTINUE`` - continue event
         processing normally.
       * ``sqlalchemy.orm.interfaces.EXT_STOP`` - cancel all subsequent
         event handlers in the chain.
       * other values - the return value specified by specific listeners.

    """
    _target_class_doc = 'SomeClass'
    _dispatch_target = mapperlib.Mapper

    @classmethod
    def _new_mapper_instance(cls, class_: Union[DeclarativeAttributeIntercept, DeclarativeMeta, type], mapper: Mapper[_O]) -> None:
        _MapperEventsHold.populate(class_, mapper)

    @classmethod
    @util.preload_module('sqlalchemy.orm')
    def _accept_with(cls, target: Union[mapperlib.Mapper[Any], Type[mapperlib.Mapper[Any]]], identifier: str) -> Optional[Union[mapperlib.Mapper[Any], Type[mapperlib.Mapper[Any]]]]:
        orm = util.preloaded.orm
        if target is orm.mapper:
            util.warn_deprecated("The `sqlalchemy.orm.mapper()` symbol is deprecated and will be removed in a future release. For the mapper-wide event target, use the 'sqlalchemy.orm.Mapper' class.", '2.0')
            return mapperlib.Mapper
        elif isinstance(target, type):
            if issubclass(target, mapperlib.Mapper):
                return target
            else:
                mapper = _mapper_or_none(target)
                if mapper is not None:
                    return mapper
                else:
                    return _MapperEventsHold(target)
        else:
            return target

    @classmethod
    def _listen(cls, event_key: _EventKey[_ET], raw: bool=False, retval: bool=False, propagate: bool=False, **kw: Any) -> None:
        target, identifier, fn = (event_key.dispatch_target, event_key.identifier, event_key._listen_fn)
        if identifier in ('before_configured', 'after_configured') and target is not mapperlib.Mapper:
            util.warn("'before_configured' and 'after_configured' ORM events only invoke with the Mapper class as the target.")
        if not raw or not retval:
            if not raw:
                meth = getattr(cls, identifier)
                try:
                    target_index = inspect_getfullargspec(meth)[0].index('target') - 1
                except ValueError:
                    target_index = None

            def wrap(*arg: Any, **kw: Any) -> Any:
                if not raw and target_index is not None:
                    arg = list(arg)
                    arg[target_index] = arg[target_index].obj()
                if not retval:
                    fn(*arg, **kw)
                    return interfaces.EXT_CONTINUE
                else:
                    return fn(*arg, **kw)
            event_key = event_key.with_wrapper(wrap)
        if propagate:
            for mapper in target.self_and_descendants:
                event_key.with_dispatch_target(mapper).base_listen(propagate=True, **kw)
        else:
            event_key.base_listen(**kw)

    @classmethod
    def _clear(cls) -> None:
        super()._clear()
        _MapperEventsHold._clear()

    def instrument_class(self, mapper: Mapper[_O], class_: Type[_O]) -> None:
        """Receive a class when the mapper is first constructed,
        before instrumentation is applied to the mapped class.

        This event is the earliest phase of mapper construction.
        Most attributes of the mapper are not yet initialized.   To
        receive an event within initial mapper construction where basic
        state is available such as the :attr:`_orm.Mapper.attrs` collection,
        the :meth:`_orm.MapperEvents.after_mapper_constructed` event may
        be a better choice.

        This listener can either be applied to the :class:`_orm.Mapper`
        class overall, or to any un-mapped class which serves as a base
        for classes that will be mapped (using the ``propagate=True`` flag)::

            Base = declarative_base()

            @event.listens_for(Base, "instrument_class", propagate=True)
            def on_new_class(mapper, cls_):
                " ... "

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param class\\_: the mapped class.

        .. seealso::

            :meth:`_orm.MapperEvents.after_mapper_constructed`

        """

    def after_mapper_constructed(self, mapper: Mapper[_O], class_: Type[_O]) -> None:
        """Receive a class and mapper when the :class:`_orm.Mapper` has been
        fully constructed.

        This event is called after the initial constructor for
        :class:`_orm.Mapper` completes.  This occurs after the
        :meth:`_orm.MapperEvents.instrument_class` event and after the
        :class:`_orm.Mapper` has done an initial pass of its arguments
        to generate its collection of :class:`_orm.MapperProperty` objects,
        which are accessible via the :meth:`_orm.Mapper.get_property`
        method and the :attr:`_orm.Mapper.iterate_properties` attribute.

        This event differs from the
        :meth:`_orm.MapperEvents.before_mapper_configured` event in that it
        is invoked within the constructor for :class:`_orm.Mapper`, rather
        than within the :meth:`_orm.registry.configure` process.   Currently,
        this event is the only one which is appropriate for handlers that
        wish to create additional mapped classes in response to the
        construction of this :class:`_orm.Mapper`, which will be part of the
        same configure step when :meth:`_orm.registry.configure` next runs.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`examples_versioning` - an example which illustrates the use
            of the :meth:`_orm.MapperEvents.before_mapper_configured`
            event to create new mappers to record change-audit histories on
            objects.

        """

    def before_mapper_configured(self, mapper: Mapper[_O], class_: Type[_O]) -> None:
        """Called right before a specific mapper is to be configured.

        This event is intended to allow a specific mapper to be skipped during
        the configure step, by returning the :attr:`.orm.interfaces.EXT_SKIP`
        symbol which indicates to the :func:`.configure_mappers` call that this
        particular mapper (or hierarchy of mappers, if ``propagate=True`` is
        used) should be skipped in the current configuration run. When one or
        more mappers are skipped, the he "new mappers" flag will remain set,
        meaning the :func:`.configure_mappers` function will continue to be
        called when mappers are used, to continue to try to configure all
        available mappers.

        In comparison to the other configure-level events,
        :meth:`.MapperEvents.before_configured`,
        :meth:`.MapperEvents.after_configured`, and
        :meth:`.MapperEvents.mapper_configured`, the
        :meth;`.MapperEvents.before_mapper_configured` event provides for a
        meaningful return value when it is registered with the ``retval=True``
        parameter.

        .. versionadded:: 1.3

        e.g.::

            from sqlalchemy.orm import EXT_SKIP

            Base = declarative_base()

            DontConfigureBase = declarative_base()

            @event.listens_for(
                DontConfigureBase,
                "before_mapper_configured", retval=True, propagate=True)
            def dont_configure(mapper, cls):
                return EXT_SKIP


        .. seealso::

            :meth:`.MapperEvents.before_configured`

            :meth:`.MapperEvents.after_configured`

            :meth:`.MapperEvents.mapper_configured`

        """

    def mapper_configured(self, mapper: Mapper[_O], class_: Type[_O]) -> None:
        """Called when a specific mapper has completed its own configuration
        within the scope of the :func:`.configure_mappers` call.

        The :meth:`.MapperEvents.mapper_configured` event is invoked
        for each mapper that is encountered when the
        :func:`_orm.configure_mappers` function proceeds through the current
        list of not-yet-configured mappers.
        :func:`_orm.configure_mappers` is typically invoked
        automatically as mappings are first used, as well as each time
        new mappers have been made available and new mapper use is
        detected.

        When the event is called, the mapper should be in its final
        state, but **not including backrefs** that may be invoked from
        other mappers; they might still be pending within the
        configuration operation.    Bidirectional relationships that
        are instead configured via the
        :paramref:`.orm.relationship.back_populates` argument
        *will* be fully available, since this style of relationship does not
        rely upon other possibly-not-configured mappers to know that they
        exist.

        For an event that is guaranteed to have **all** mappers ready
        to go including backrefs that are defined only on other
        mappings, use the :meth:`.MapperEvents.after_configured`
        event; this event invokes only after all known mappings have been
        fully configured.

        The :meth:`.MapperEvents.mapper_configured` event, unlike
        :meth:`.MapperEvents.before_configured` or
        :meth:`.MapperEvents.after_configured`,
        is called for each mapper/class individually, and the mapper is
        passed to the event itself.  It also is called exactly once for
        a particular mapper.  The event is therefore useful for
        configurational steps that benefit from being invoked just once
        on a specific mapper basis, which don't require that "backref"
        configurations are necessarily ready yet.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param class\\_: the mapped class.

        .. seealso::

            :meth:`.MapperEvents.before_configured`

            :meth:`.MapperEvents.after_configured`

            :meth:`.MapperEvents.before_mapper_configured`

        """

    def before_configured(self) -> None:
        """Called before a series of mappers have been configured.

        The :meth:`.MapperEvents.before_configured` event is invoked
        each time the :func:`_orm.configure_mappers` function is
        invoked, before the function has done any of its work.
        :func:`_orm.configure_mappers` is typically invoked
        automatically as mappings are first used, as well as each time
        new mappers have been made available and new mapper use is
        detected.

        This event can **only** be applied to the :class:`_orm.Mapper` class,
        and not to individual mappings or mapped classes. It is only invoked
        for all mappings as a whole::

            from sqlalchemy.orm import Mapper

            @event.listens_for(Mapper, "before_configured")
            def go():
                ...

        Contrast this event to :meth:`.MapperEvents.after_configured`,
        which is invoked after the series of mappers has been configured,
        as well as :meth:`.MapperEvents.before_mapper_configured`
        and :meth:`.MapperEvents.mapper_configured`, which are both invoked
        on a per-mapper basis.

        Theoretically this event is called once per
        application, but is actually called any time new mappers
        are to be affected by a :func:`_orm.configure_mappers`
        call.   If new mappings are constructed after existing ones have
        already been used, this event will likely be called again.  To ensure
        that a particular event is only called once and no further, the
        ``once=True`` argument (new in 0.9.4) can be applied::

            from sqlalchemy.orm import mapper

            @event.listens_for(mapper, "before_configured", once=True)
            def go():
                ...


        .. seealso::

            :meth:`.MapperEvents.before_mapper_configured`

            :meth:`.MapperEvents.mapper_configured`

            :meth:`.MapperEvents.after_configured`

        """

    def after_configured(self) -> None:
        """Called after a series of mappers have been configured.

        The :meth:`.MapperEvents.after_configured` event is invoked
        each time the :func:`_orm.configure_mappers` function is
        invoked, after the function has completed its work.
        :func:`_orm.configure_mappers` is typically invoked
        automatically as mappings are first used, as well as each time
        new mappers have been made available and new mapper use is
        detected.

        Contrast this event to the :meth:`.MapperEvents.mapper_configured`
        event, which is called on a per-mapper basis while the configuration
        operation proceeds; unlike that event, when this event is invoked,
        all cross-configurations (e.g. backrefs) will also have been made
        available for any mappers that were pending.
        Also contrast to :meth:`.MapperEvents.before_configured`,
        which is invoked before the series of mappers has been configured.

        This event can **only** be applied to the :class:`_orm.Mapper` class,
        and not to individual mappings or
        mapped classes.  It is only invoked for all mappings as a whole::

            from sqlalchemy.orm import Mapper

            @event.listens_for(Mapper, "after_configured")
            def go():
                # ...

        Theoretically this event is called once per
        application, but is actually called any time new mappers
        have been affected by a :func:`_orm.configure_mappers`
        call.   If new mappings are constructed after existing ones have
        already been used, this event will likely be called again.  To ensure
        that a particular event is only called once and no further, the
        ``once=True`` argument (new in 0.9.4) can be applied::

            from sqlalchemy.orm import mapper

            @event.listens_for(mapper, "after_configured", once=True)
            def go():
                # ...

        .. seealso::

            :meth:`.MapperEvents.before_mapper_configured`

            :meth:`.MapperEvents.mapper_configured`

            :meth:`.MapperEvents.before_configured`

        """

    def before_insert(self, mapper: Mapper[_O], connection: Connection, target: _O) -> None:
        """Receive an object instance before an INSERT statement
        is emitted corresponding to that instance.

        .. note:: this event **only** applies to the
           :ref:`session flush operation <session_flushing>`
           and does **not** apply to the ORM DML operations described at
           :ref:`orm_expression_update_delete`.  To intercept ORM
           DML events, use :meth:`_orm.SessionEvents.do_orm_execute`.

        This event is used to modify local, non-object related
        attributes on the instance before an INSERT occurs, as well
        as to emit additional SQL statements on the given
        connection.

        The event is often called for a batch of objects of the
        same class before their INSERT statements are emitted at
        once in a later step. In the extremely rare case that
        this is not desirable, the :class:`_orm.Mapper` object can be
        configured with ``batch=False``, which will cause
        batches of instances to be broken up into individual
        (and more poorly performing) event->persist->event
        steps.

        .. warning::

            Mapper-level flush events only allow **very limited operations**,
            on attributes local to the row being operated upon only,
            as well as allowing any SQL to be emitted on the given
            :class:`_engine.Connection`.  **Please read fully** the notes
            at :ref:`session_persistence_mapper` for guidelines on using
            these methods; generally, the :meth:`.SessionEvents.before_flush`
            method should be preferred for general on-flush changes.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param connection: the :class:`_engine.Connection` being used to
         emit INSERT statements for this instance.  This
         provides a handle into the current transaction on the
         target database specific to this instance.
        :param target: the mapped instance being persisted.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :return: No return value is supported by this event.

        .. seealso::

            :ref:`session_persistence_events`

        """

    def after_insert(self, mapper: Mapper[_O], connection: Connection, target: _O) -> None:
        """Receive an object instance after an INSERT statement
        is emitted corresponding to that instance.

        .. note:: this event **only** applies to the
           :ref:`session flush operation <session_flushing>`
           and does **not** apply to the ORM DML operations described at
           :ref:`orm_expression_update_delete`.  To intercept ORM
           DML events, use :meth:`_orm.SessionEvents.do_orm_execute`.

        This event is used to modify in-Python-only
        state on the instance after an INSERT occurs, as well
        as to emit additional SQL statements on the given
        connection.

        The event is often called for a batch of objects of the
        same class after their INSERT statements have been
        emitted at once in a previous step. In the extremely
        rare case that this is not desirable, the
        :class:`_orm.Mapper` object can be configured with ``batch=False``,
        which will cause batches of instances to be broken up
        into individual (and more poorly performing)
        event->persist->event steps.

        .. warning::

            Mapper-level flush events only allow **very limited operations**,
            on attributes local to the row being operated upon only,
            as well as allowing any SQL to be emitted on the given
            :class:`_engine.Connection`.  **Please read fully** the notes
            at :ref:`session_persistence_mapper` for guidelines on using
            these methods; generally, the :meth:`.SessionEvents.before_flush`
            method should be preferred for general on-flush changes.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param connection: the :class:`_engine.Connection` being used to
         emit INSERT statements for this instance.  This
         provides a handle into the current transaction on the
         target database specific to this instance.
        :param target: the mapped instance being persisted.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :return: No return value is supported by this event.

        .. seealso::

            :ref:`session_persistence_events`

        """

    def before_update(self, mapper: Mapper[_O], connection: Connection, target: _O) -> None:
        """Receive an object instance before an UPDATE statement
        is emitted corresponding to that instance.

        .. note:: this event **only** applies to the
           :ref:`session flush operation <session_flushing>`
           and does **not** apply to the ORM DML operations described at
           :ref:`orm_expression_update_delete`.  To intercept ORM
           DML events, use :meth:`_orm.SessionEvents.do_orm_execute`.

        This event is used to modify local, non-object related
        attributes on the instance before an UPDATE occurs, as well
        as to emit additional SQL statements on the given
        connection.

        This method is called for all instances that are
        marked as "dirty", *even those which have no net changes
        to their column-based attributes*. An object is marked
        as dirty when any of its column-based attributes have a
        "set attribute" operation called or when any of its
        collections are modified. If, at update time, no
        column-based attributes have any net changes, no UPDATE
        statement will be issued. This means that an instance
        being sent to :meth:`~.MapperEvents.before_update` is
        *not* a guarantee that an UPDATE statement will be
        issued, although you can affect the outcome here by
        modifying attributes so that a net change in value does
        exist.

        To detect if the column-based attributes on the object have net
        changes, and will therefore generate an UPDATE statement, use
        ``object_session(instance).is_modified(instance,
        include_collections=False)``.

        The event is often called for a batch of objects of the
        same class before their UPDATE statements are emitted at
        once in a later step. In the extremely rare case that
        this is not desirable, the :class:`_orm.Mapper` can be
        configured with ``batch=False``, which will cause
        batches of instances to be broken up into individual
        (and more poorly performing) event->persist->event
        steps.

        .. warning::

            Mapper-level flush events only allow **very limited operations**,
            on attributes local to the row being operated upon only,
            as well as allowing any SQL to be emitted on the given
            :class:`_engine.Connection`.  **Please read fully** the notes
            at :ref:`session_persistence_mapper` for guidelines on using
            these methods; generally, the :meth:`.SessionEvents.before_flush`
            method should be preferred for general on-flush changes.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param connection: the :class:`_engine.Connection` being used to
         emit UPDATE statements for this instance.  This
         provides a handle into the current transaction on the
         target database specific to this instance.
        :param target: the mapped instance being persisted.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :return: No return value is supported by this event.

        .. seealso::

            :ref:`session_persistence_events`

        """

    def after_update(self, mapper: Mapper[_O], connection: Connection, target: _O) -> None:
        """Receive an object instance after an UPDATE statement
        is emitted corresponding to that instance.

        .. note:: this event **only** applies to the
           :ref:`session flush operation <session_flushing>`
           and does **not** apply to the ORM DML operations described at
           :ref:`orm_expression_update_delete`.  To intercept ORM
           DML events, use :meth:`_orm.SessionEvents.do_orm_execute`.

        This event is used to modify in-Python-only
        state on the instance after an UPDATE occurs, as well
        as to emit additional SQL statements on the given
        connection.

        This method is called for all instances that are
        marked as "dirty", *even those which have no net changes
        to their column-based attributes*, and for which
        no UPDATE statement has proceeded. An object is marked
        as dirty when any of its column-based attributes have a
        "set attribute" operation called or when any of its
        collections are modified. If, at update time, no
        column-based attributes have any net changes, no UPDATE
        statement will be issued. This means that an instance
        being sent to :meth:`~.MapperEvents.after_update` is
        *not* a guarantee that an UPDATE statement has been
        issued.

        To detect if the column-based attributes on the object have net
        changes, and therefore resulted in an UPDATE statement, use
        ``object_session(instance).is_modified(instance,
        include_collections=False)``.

        The event is often called for a batch of objects of the
        same class after their UPDATE statements have been emitted at
        once in a previous step. In the extremely rare case that
        this is not desirable, the :class:`_orm.Mapper` can be
        configured with ``batch=False``, which will cause
        batches of instances to be broken up into individual
        (and more poorly performing) event->persist->event
        steps.

        .. warning::

            Mapper-level flush events only allow **very limited operations**,
            on attributes local to the row being operated upon only,
            as well as allowing any SQL to be emitted on the given
            :class:`_engine.Connection`.  **Please read fully** the notes
            at :ref:`session_persistence_mapper` for guidelines on using
            these methods; generally, the :meth:`.SessionEvents.before_flush`
            method should be preferred for general on-flush changes.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param connection: the :class:`_engine.Connection` being used to
         emit UPDATE statements for this instance.  This
         provides a handle into the current transaction on the
         target database specific to this instance.
        :param target: the mapped instance being persisted.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :return: No return value is supported by this event.

        .. seealso::

            :ref:`session_persistence_events`

        """

    def before_delete(self, mapper: Mapper[_O], connection: Connection, target: _O) -> None:
        """Receive an object instance before a DELETE statement
        is emitted corresponding to that instance.

        .. note:: this event **only** applies to the
           :ref:`session flush operation <session_flushing>`
           and does **not** apply to the ORM DML operations described at
           :ref:`orm_expression_update_delete`.  To intercept ORM
           DML events, use :meth:`_orm.SessionEvents.do_orm_execute`.

        This event is used to emit additional SQL statements on
        the given connection as well as to perform application
        specific bookkeeping related to a deletion event.

        The event is often called for a batch of objects of the
        same class before their DELETE statements are emitted at
        once in a later step.

        .. warning::

            Mapper-level flush events only allow **very limited operations**,
            on attributes local to the row being operated upon only,
            as well as allowing any SQL to be emitted on the given
            :class:`_engine.Connection`.  **Please read fully** the notes
            at :ref:`session_persistence_mapper` for guidelines on using
            these methods; generally, the :meth:`.SessionEvents.before_flush`
            method should be preferred for general on-flush changes.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param connection: the :class:`_engine.Connection` being used to
         emit DELETE statements for this instance.  This
         provides a handle into the current transaction on the
         target database specific to this instance.
        :param target: the mapped instance being deleted.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :return: No return value is supported by this event.

        .. seealso::

            :ref:`session_persistence_events`

        """

    def after_delete(self, mapper: Mapper[_O], connection: Connection, target: _O) -> None:
        """Receive an object instance after a DELETE statement
        has been emitted corresponding to that instance.

        .. note:: this event **only** applies to the
           :ref:`session flush operation <session_flushing>`
           and does **not** apply to the ORM DML operations described at
           :ref:`orm_expression_update_delete`.  To intercept ORM
           DML events, use :meth:`_orm.SessionEvents.do_orm_execute`.

        This event is used to emit additional SQL statements on
        the given connection as well as to perform application
        specific bookkeeping related to a deletion event.

        The event is often called for a batch of objects of the
        same class after their DELETE statements have been emitted at
        once in a previous step.

        .. warning::

            Mapper-level flush events only allow **very limited operations**,
            on attributes local to the row being operated upon only,
            as well as allowing any SQL to be emitted on the given
            :class:`_engine.Connection`.  **Please read fully** the notes
            at :ref:`session_persistence_mapper` for guidelines on using
            these methods; generally, the :meth:`.SessionEvents.before_flush`
            method should be preferred for general on-flush changes.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param connection: the :class:`_engine.Connection` being used to
         emit DELETE statements for this instance.  This
         provides a handle into the current transaction on the
         target database specific to this instance.
        :param target: the mapped instance being deleted.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :return: No return value is supported by this event.

        .. seealso::

            :ref:`session_persistence_events`

        """