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
class AttributeEvents(event.Events[QueryableAttribute[Any]]):
    """Define events for object attributes.

    These are typically defined on the class-bound descriptor for the
    target class.

    For example, to register a listener that will receive the
    :meth:`_orm.AttributeEvents.append` event::

        from sqlalchemy import event

        @event.listens_for(MyClass.collection, 'append', propagate=True)
        def my_append_listener(target, value, initiator):
            print("received append event for target: %s" % target)


    Listeners have the option to return a possibly modified version of the
    value, when the :paramref:`.AttributeEvents.retval` flag is passed to
    :func:`.event.listen` or :func:`.event.listens_for`, such as below,
    illustrated using the :meth:`_orm.AttributeEvents.set` event::

        def validate_phone(target, value, oldvalue, initiator):
            "Strip non-numeric characters from a phone number"

            return re.sub(r'\\D', '', value)

        # setup listener on UserContact.phone attribute, instructing
        # it to use the return value
        listen(UserContact.phone, 'set', validate_phone, retval=True)

    A validation function like the above can also raise an exception
    such as :exc:`ValueError` to halt the operation.

    The :paramref:`.AttributeEvents.propagate` flag is also important when
    applying listeners to mapped classes that also have mapped subclasses,
    as when using mapper inheritance patterns::


        @event.listens_for(MySuperClass.attr, 'set', propagate=True)
        def receive_set(target, value, initiator):
            print("value set: %s" % target)

    The full list of modifiers available to the :func:`.event.listen`
    and :func:`.event.listens_for` functions are below.

    :param active_history=False: When True, indicates that the
      "set" event would like to receive the "old" value being
      replaced unconditionally, even if this requires firing off
      database loads. Note that ``active_history`` can also be
      set directly via :func:`.column_property` and
      :func:`_orm.relationship`.

    :param propagate=False: When True, the listener function will
      be established not just for the class attribute given, but
      for attributes of the same name on all current subclasses
      of that class, as well as all future subclasses of that
      class, using an additional listener that listens for
      instrumentation events.
    :param raw=False: When True, the "target" argument to the
      event will be the :class:`.InstanceState` management
      object, rather than the mapped instance itself.
    :param retval=False: when True, the user-defined event
      listening must return the "value" argument from the
      function.  This gives the listening function the opportunity
      to change the value that is ultimately used for a "set"
      or "append" event.

    """
    _target_class_doc = 'SomeClass.some_attribute'
    _dispatch_target = QueryableAttribute

    @staticmethod
    def _set_dispatch(cls: Type[_HasEventsDispatch[Any]], dispatch_cls: Type[_Dispatch[Any]]) -> _Dispatch[Any]:
        dispatch = event.Events._set_dispatch(cls, dispatch_cls)
        dispatch_cls._active_history = False
        return dispatch

    @classmethod
    def _accept_with(cls, target: Union[QueryableAttribute[Any], Type[QueryableAttribute[Any]]], identifier: str) -> Union[QueryableAttribute[Any], Type[QueryableAttribute[Any]]]:
        if isinstance(target, interfaces.MapperProperty):
            return getattr(target.parent.class_, target.key)
        else:
            return target

    @classmethod
    def _listen(cls, event_key: _EventKey[QueryableAttribute[Any]], active_history: bool=False, raw: bool=False, retval: bool=False, propagate: bool=False, include_key: bool=False) -> None:
        target, fn = (event_key.dispatch_target, event_key._listen_fn)
        if active_history:
            target.dispatch._active_history = True
        if not raw or not retval or (not include_key):

            def wrap(target: InstanceState[_O], *arg: Any, **kw: Any) -> Any:
                if not raw:
                    target = target.obj()
                if not retval:
                    if arg:
                        value = arg[0]
                    else:
                        value = None
                    if include_key:
                        fn(target, *arg, **kw)
                    else:
                        fn(target, *arg)
                    return value
                elif include_key:
                    return fn(target, *arg, **kw)
                else:
                    return fn(target, *arg)
            event_key = event_key.with_wrapper(wrap)
        event_key.base_listen(propagate=propagate)
        if propagate:
            manager = instrumentation.manager_of_class(target.class_)
            for mgr in manager.subclass_managers(True):
                event_key.with_dispatch_target(mgr[target.key]).base_listen(propagate=True)
                if active_history:
                    mgr[target.key].dispatch._active_history = True

    def append(self, target: _O, value: _T, initiator: Event, *, key: EventConstants=NO_KEY) -> Optional[_T]:
        """Receive a collection append event.

        The append event is invoked for each element as it is appended
        to the collection.  This occurs for single-item appends as well
        as for a "bulk replace" operation.

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.
        :param value: the value being appended.  If this listener
          is registered with ``retval=True``, the listener
          function must return this value, or a new value which
          replaces it.
        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.  May be modified
          from its original value by backref handlers in order to control
          chained event propagation, as well as be inspected for information
          about the source of the event.
        :param key: When the event is established using the
         :paramref:`.AttributeEvents.include_key` parameter set to
         True, this will be the key used in the operation, such as
         ``collection[some_key_or_index] = value``.
         The parameter is not passed
         to the event at all if the the
         :paramref:`.AttributeEvents.include_key`
         was not used to set up the event; this is to allow backwards
         compatibility with existing event handlers that don't include the
         ``key`` parameter.

         .. versionadded:: 2.0

        :return: if the event was registered with ``retval=True``,
         the given value, or a new effective value, should be returned.

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

            :meth:`.AttributeEvents.bulk_replace`

        """

    def append_wo_mutation(self, target: _O, value: _T, initiator: Event, *, key: EventConstants=NO_KEY) -> None:
        """Receive a collection append event where the collection was not
        actually mutated.

        This event differs from :meth:`_orm.AttributeEvents.append` in that
        it is fired off for de-duplicating collections such as sets and
        dictionaries, when the object already exists in the target collection.
        The event does not have a return value and the identity of the
        given object cannot be changed.

        The event is used for cascading objects into a :class:`_orm.Session`
        when the collection has already been mutated via a backref event.

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.
        :param value: the value that would be appended if the object did not
          already exist in the collection.
        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.  May be modified
          from its original value by backref handlers in order to control
          chained event propagation, as well as be inspected for information
          about the source of the event.
        :param key: When the event is established using the
         :paramref:`.AttributeEvents.include_key` parameter set to
         True, this will be the key used in the operation, such as
         ``collection[some_key_or_index] = value``.
         The parameter is not passed
         to the event at all if the the
         :paramref:`.AttributeEvents.include_key`
         was not used to set up the event; this is to allow backwards
         compatibility with existing event handlers that don't include the
         ``key`` parameter.

         .. versionadded:: 2.0

        :return: No return value is defined for this event.

        .. versionadded:: 1.4.15

        """

    def bulk_replace(self, target: _O, values: Iterable[_T], initiator: Event, *, keys: Optional[Iterable[EventConstants]]=None) -> None:
        """Receive a collection 'bulk replace' event.

        This event is invoked for a sequence of values as they are incoming
        to a bulk collection set operation, which can be
        modified in place before the values are treated as ORM objects.
        This is an "early hook" that runs before the bulk replace routine
        attempts to reconcile which objects are already present in the
        collection and which are being removed by the net replace operation.

        It is typical that this method be combined with use of the
        :meth:`.AttributeEvents.append` event.    When using both of these
        events, note that a bulk replace operation will invoke
        the :meth:`.AttributeEvents.append` event for all new items,
        even after :meth:`.AttributeEvents.bulk_replace` has been invoked
        for the collection as a whole.  In order to determine if an
        :meth:`.AttributeEvents.append` event is part of a bulk replace,
        use the symbol :attr:`~.attributes.OP_BULK_REPLACE` to test the
        incoming initiator::

            from sqlalchemy.orm.attributes import OP_BULK_REPLACE

            @event.listens_for(SomeObject.collection, "bulk_replace")
            def process_collection(target, values, initiator):
                values[:] = [_make_value(value) for value in values]

            @event.listens_for(SomeObject.collection, "append", retval=True)
            def process_collection(target, value, initiator):
                # make sure bulk_replace didn't already do it
                if initiator is None or initiator.op is not OP_BULK_REPLACE:
                    return _make_value(value)
                else:
                    return value

        .. versionadded:: 1.2

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.
        :param value: a sequence (e.g. a list) of the values being set.  The
          handler can modify this list in place.
        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.
        :param keys: When the event is established using the
         :paramref:`.AttributeEvents.include_key` parameter set to
         True, this will be the sequence of keys used in the operation,
         typically only for a dictionary update.  The parameter is not passed
         to the event at all if the the
         :paramref:`.AttributeEvents.include_key`
         was not used to set up the event; this is to allow backwards
         compatibility with existing event handlers that don't include the
         ``key`` parameter.

         .. versionadded:: 2.0

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.


        """

    def remove(self, target: _O, value: _T, initiator: Event, *, key: EventConstants=NO_KEY) -> None:
        """Receive a collection remove event.

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.
        :param value: the value being removed.
        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.  May be modified
          from its original value by backref handlers in order to control
          chained event propagation.

        :param key: When the event is established using the
         :paramref:`.AttributeEvents.include_key` parameter set to
         True, this will be the key used in the operation, such as
         ``del collection[some_key_or_index]``.  The parameter is not passed
         to the event at all if the the
         :paramref:`.AttributeEvents.include_key`
         was not used to set up the event; this is to allow backwards
         compatibility with existing event handlers that don't include the
         ``key`` parameter.

         .. versionadded:: 2.0

        :return: No return value is defined for this event.


        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

        """

    def set(self, target: _O, value: _T, oldvalue: _T, initiator: Event) -> None:
        """Receive a scalar set event.

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.
        :param value: the value being set.  If this listener
          is registered with ``retval=True``, the listener
          function must return this value, or a new value which
          replaces it.
        :param oldvalue: the previous value being replaced.  This
          may also be the symbol ``NEVER_SET`` or ``NO_VALUE``.
          If the listener is registered with ``active_history=True``,
          the previous value of the attribute will be loaded from
          the database if the existing value is currently unloaded
          or expired.
        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.  May be modified
          from its original value by backref handlers in order to control
          chained event propagation.

        :return: if the event was registered with ``retval=True``,
         the given value, or a new effective value, should be returned.

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

        """

    def init_scalar(self, target: _O, value: _T, dict_: Dict[Any, Any]) -> None:
        """Receive a scalar "init" event.

        This event is invoked when an uninitialized, unpersisted scalar
        attribute is accessed, e.g. read::


            x = my_object.some_attribute

        The ORM's default behavior when this occurs for an un-initialized
        attribute is to return the value ``None``; note this differs from
        Python's usual behavior of raising ``AttributeError``.    The
        event here can be used to customize what value is actually returned,
        with the assumption that the event listener would be mirroring
        a default generator that is configured on the Core
        :class:`_schema.Column`
        object as well.

        Since a default generator on a :class:`_schema.Column`
        might also produce
        a changing value such as a timestamp, the
        :meth:`.AttributeEvents.init_scalar`
        event handler can also be used to **set** the newly returned value, so
        that a Core-level default generation function effectively fires off
        only once, but at the moment the attribute is accessed on the
        non-persisted object.   Normally, no change to the object's state
        is made when an uninitialized attribute is accessed (much older
        SQLAlchemy versions did in fact change the object's state).

        If a default generator on a column returned a particular constant,
        a handler might be used as follows::

            SOME_CONSTANT = 3.1415926

            class MyClass(Base):
                # ...

                some_attribute = Column(Numeric, default=SOME_CONSTANT)

            @event.listens_for(
                MyClass.some_attribute, "init_scalar",
                retval=True, propagate=True)
            def _init_some_attribute(target, dict_, value):
                dict_['some_attribute'] = SOME_CONSTANT
                return SOME_CONSTANT

        Above, we initialize the attribute ``MyClass.some_attribute`` to the
        value of ``SOME_CONSTANT``.   The above code includes the following
        features:

        * By setting the value ``SOME_CONSTANT`` in the given ``dict_``,
          we indicate that this value is to be persisted to the database.
          This supersedes the use of ``SOME_CONSTANT`` in the default generator
          for the :class:`_schema.Column`.  The ``active_column_defaults.py``
          example given at :ref:`examples_instrumentation` illustrates using
          the same approach for a changing default, e.g. a timestamp
          generator.    In this particular example, it is not strictly
          necessary to do this since ``SOME_CONSTANT`` would be part of the
          INSERT statement in either case.

        * By establishing the ``retval=True`` flag, the value we return
          from the function will be returned by the attribute getter.
          Without this flag, the event is assumed to be a passive observer
          and the return value of our function is ignored.

        * The ``propagate=True`` flag is significant if the mapped class
          includes inheriting subclasses, which would also make use of this
          event listener.  Without this flag, an inheriting subclass will
          not use our event handler.

        In the above example, the attribute set event
        :meth:`.AttributeEvents.set` as well as the related validation feature
        provided by :obj:`_orm.validates` is **not** invoked when we apply our
        value to the given ``dict_``.  To have these events to invoke in
        response to our newly generated value, apply the value to the given
        object as a normal attribute set operation::

            SOME_CONSTANT = 3.1415926

            @event.listens_for(
                MyClass.some_attribute, "init_scalar",
                retval=True, propagate=True)
            def _init_some_attribute(target, dict_, value):
                # will also fire off attribute set events
                target.some_attribute = SOME_CONSTANT
                return SOME_CONSTANT

        When multiple listeners are set up, the generation of the value
        is "chained" from one listener to the next by passing the value
        returned by the previous listener that specifies ``retval=True``
        as the ``value`` argument of the next listener.

        :param target: the object instance receiving the event.
         If the listener is registered with ``raw=True``, this will
         be the :class:`.InstanceState` object.
        :param value: the value that is to be returned before this event
         listener were invoked.  This value begins as the value ``None``,
         however will be the return value of the previous event handler
         function if multiple listeners are present.
        :param dict\\_: the attribute dictionary of this mapped object.
         This is normally the ``__dict__`` of the object, but in all cases
         represents the destination that the attribute system uses to get
         at the actual value of this attribute.  Placing the value in this
         dictionary has the effect that the value will be used in the
         INSERT statement generated by the unit of work.


        .. seealso::

            :meth:`.AttributeEvents.init_collection` - collection version
            of this event

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

            :ref:`examples_instrumentation` - see the
            ``active_column_defaults.py`` example.

        """

    def init_collection(self, target: _O, collection: Type[Collection[Any]], collection_adapter: CollectionAdapter) -> None:
        """Receive a 'collection init' event.

        This event is triggered for a collection-based attribute, when
        the initial "empty collection" is first generated for a blank
        attribute, as well as for when the collection is replaced with
        a new one, such as via a set event.

        E.g., given that ``User.addresses`` is a relationship-based
        collection, the event is triggered here::

            u1 = User()
            u1.addresses.append(a1)  #  <- new collection

        and also during replace operations::

            u1.addresses = [a2, a3]  #  <- new collection

        :param target: the object instance receiving the event.
         If the listener is registered with ``raw=True``, this will
         be the :class:`.InstanceState` object.
        :param collection: the new collection.  This will always be generated
         from what was specified as
         :paramref:`_orm.relationship.collection_class`, and will always
         be empty.
        :param collection_adapter: the :class:`.CollectionAdapter` that will
         mediate internal access to the collection.

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

            :meth:`.AttributeEvents.init_scalar` - "scalar" version of this
            event.

        """

    def dispose_collection(self, target: _O, collection: Collection[Any], collection_adapter: CollectionAdapter) -> None:
        """Receive a 'collection dispose' event.

        This event is triggered for a collection-based attribute when
        a collection is replaced, that is::

            u1.addresses.append(a1)

            u1.addresses = [a2, a3]  # <- old collection is disposed

        The old collection received will contain its previous contents.

        .. versionchanged:: 1.2 The collection passed to
           :meth:`.AttributeEvents.dispose_collection` will now have its
           contents before the dispose intact; previously, the collection
           would be empty.

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

        """

    def modified(self, target: _O, initiator: Event) -> None:
        """Receive a 'modified' event.

        This event is triggered when the :func:`.attributes.flag_modified`
        function is used to trigger a modify event on an attribute without
        any specific value being set.

        .. versionadded:: 1.2

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.

        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

        """