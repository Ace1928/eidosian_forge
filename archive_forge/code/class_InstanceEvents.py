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
class InstanceEvents(event.Events[ClassManager[Any]]):
    """Define events specific to object lifecycle.

    e.g.::

        from sqlalchemy import event

        def my_load_listener(target, context):
            print("on load!")

        event.listen(SomeClass, 'load', my_load_listener)

    Available targets include:

    * mapped classes
    * unmapped superclasses of mapped or to-be-mapped classes
      (using the ``propagate=True`` flag)
    * :class:`_orm.Mapper` objects
    * the :class:`_orm.Mapper` class itself indicates listening for all
      mappers.

    Instance events are closely related to mapper events, but
    are more specific to the instance and its instrumentation,
    rather than its system of persistence.

    When using :class:`.InstanceEvents`, several modifiers are
    available to the :func:`.event.listen` function.

    :param propagate=False: When True, the event listener should
       be applied to all inheriting classes as well as the
       class which is the target of this listener.
    :param raw=False: When True, the "target" argument passed
       to applicable event listener functions will be the
       instance's :class:`.InstanceState` management
       object, rather than the mapped instance itself.
    :param restore_load_context=False: Applies to the
       :meth:`.InstanceEvents.load` and :meth:`.InstanceEvents.refresh`
       events.  Restores the loader context of the object when the event
       hook is complete, so that ongoing eager load operations continue
       to target the object appropriately.  A warning is emitted if the
       object is moved to a new loader context from within one of these
       events if this flag is not set.

       .. versionadded:: 1.3.14


    """
    _target_class_doc = 'SomeClass'
    _dispatch_target = ClassManager

    @classmethod
    def _new_classmanager_instance(cls, class_: Union[DeclarativeAttributeIntercept, DeclarativeMeta, type], classmanager: ClassManager[_O]) -> None:
        _InstanceEventsHold.populate(class_, classmanager)

    @classmethod
    @util.preload_module('sqlalchemy.orm')
    def _accept_with(cls, target: Union[ClassManager[Any], Type[ClassManager[Any]]], identifier: str) -> Optional[Union[ClassManager[Any], Type[ClassManager[Any]]]]:
        orm = util.preloaded.orm
        if isinstance(target, ClassManager):
            return target
        elif isinstance(target, mapperlib.Mapper):
            return target.class_manager
        elif target is orm.mapper:
            util.warn_deprecated("The `sqlalchemy.orm.mapper()` symbol is deprecated and will be removed in a future release. For the mapper-wide event target, use the 'sqlalchemy.orm.Mapper' class.", '2.0')
            return ClassManager
        elif isinstance(target, type):
            if issubclass(target, mapperlib.Mapper):
                return ClassManager
            else:
                manager = instrumentation.opt_manager_of_class(target)
                if manager:
                    return manager
                else:
                    return _InstanceEventsHold(target)
        return None

    @classmethod
    def _listen(cls, event_key: _EventKey[ClassManager[Any]], raw: bool=False, propagate: bool=False, restore_load_context: bool=False, **kw: Any) -> None:
        target, fn = (event_key.dispatch_target, event_key._listen_fn)
        if not raw or restore_load_context:

            def wrap(state: InstanceState[_O], *arg: Any, **kw: Any) -> Optional[Any]:
                if not raw:
                    target: Any = state.obj()
                else:
                    target = state
                if restore_load_context:
                    runid = state.runid
                try:
                    return fn(target, *arg, **kw)
                finally:
                    if restore_load_context:
                        state.runid = runid
            event_key = event_key.with_wrapper(wrap)
        event_key.base_listen(propagate=propagate, **kw)
        if propagate:
            for mgr in target.subclass_managers(True):
                event_key.with_dispatch_target(mgr).base_listen(propagate=True)

    @classmethod
    def _clear(cls) -> None:
        super()._clear()
        _InstanceEventsHold._clear()

    def first_init(self, manager: ClassManager[_O], cls: Type[_O]) -> None:
        """Called when the first instance of a particular mapping is called.

        This event is called when the ``__init__`` method of a class
        is called the first time for that particular class.    The event
        invokes before ``__init__`` actually proceeds as well as before
        the :meth:`.InstanceEvents.init` event is invoked.

        """

    def init(self, target: _O, args: Any, kwargs: Any) -> None:
        """Receive an instance when its constructor is called.

        This method is only called during a userland construction of
        an object, in conjunction with the object's constructor, e.g.
        its ``__init__`` method.  It is not called when an object is
        loaded from the database; see the :meth:`.InstanceEvents.load`
        event in order to intercept a database load.

        The event is called before the actual ``__init__`` constructor
        of the object is called.  The ``kwargs`` dictionary may be
        modified in-place in order to affect what is passed to
        ``__init__``.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param args: positional arguments passed to the ``__init__`` method.
         This is passed as a tuple and is currently immutable.
        :param kwargs: keyword arguments passed to the ``__init__`` method.
         This structure *can* be altered in place.

        .. seealso::

            :meth:`.InstanceEvents.init_failure`

            :meth:`.InstanceEvents.load`

        """

    def init_failure(self, target: _O, args: Any, kwargs: Any) -> None:
        """Receive an instance when its constructor has been called,
        and raised an exception.

        This method is only called during a userland construction of
        an object, in conjunction with the object's constructor, e.g.
        its ``__init__`` method. It is not called when an object is loaded
        from the database.

        The event is invoked after an exception raised by the ``__init__``
        method is caught.  After the event
        is invoked, the original exception is re-raised outwards, so that
        the construction of the object still raises an exception.   The
        actual exception and stack trace raised should be present in
        ``sys.exc_info()``.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param args: positional arguments that were passed to the ``__init__``
         method.
        :param kwargs: keyword arguments that were passed to the ``__init__``
         method.

        .. seealso::

            :meth:`.InstanceEvents.init`

            :meth:`.InstanceEvents.load`

        """

    def _sa_event_merge_wo_load(self, target: _O, context: QueryContext) -> None:
        """receive an object instance after it was the subject of a merge()
        call, when load=False was passed.

        The target would be the already-loaded object in the Session which
        would have had its attributes overwritten by the incoming object. This
        overwrite operation does not use attribute events, instead just
        populating dict directly. Therefore the purpose of this event is so
        that extensions like sqlalchemy.ext.mutable know that object state has
        changed and incoming state needs to be set up for "parents" etc.

        This functionality is acceptable to be made public in a later release.

        .. versionadded:: 1.4.41

        """

    def load(self, target: _O, context: QueryContext) -> None:
        """Receive an object instance after it has been created via
        ``__new__``, and after initial attribute population has
        occurred.

        This typically occurs when the instance is created based on
        incoming result rows, and is only called once for that
        instance's lifetime.

        .. warning::

            During a result-row load, this event is invoked when the
            first row received for this instance is processed.  When using
            eager loading with collection-oriented attributes, the additional
            rows that are to be loaded / processed in order to load subsequent
            collection items have not occurred yet.   This has the effect
            both that collections will not be fully loaded, as well as that
            if an operation occurs within this event handler that emits
            another database load operation for the object, the "loading
            context" for the object can change and interfere with the
            existing eager loaders still in progress.

            Examples of what can cause the "loading context" to change within
            the event handler include, but are not necessarily limited to:

            * accessing deferred attributes that weren't part of the row,
              will trigger an "undefer" operation and refresh the object

            * accessing attributes on a joined-inheritance subclass that
              weren't part of the row, will trigger a refresh operation.

            As of SQLAlchemy 1.3.14, a warning is emitted when this occurs. The
            :paramref:`.InstanceEvents.restore_load_context` option may  be
            used on the event to prevent this warning; this will ensure that
            the existing loading context is maintained for the object after the
            event is called::

                @event.listens_for(
                    SomeClass, "load", restore_load_context=True)
                def on_load(instance, context):
                    instance.some_unloaded_attribute

            .. versionchanged:: 1.3.14 Added
               :paramref:`.InstanceEvents.restore_load_context`
               and :paramref:`.SessionEvents.restore_load_context` flags which
               apply to "on load" events, which will ensure that the loading
               context for an object is restored when the event hook is
               complete; a warning is emitted if the load context of the object
               changes without this flag being set.


        The :meth:`.InstanceEvents.load` event is also available in a
        class-method decorator format called :func:`_orm.reconstructor`.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param context: the :class:`.QueryContext` corresponding to the
         current :class:`_query.Query` in progress.  This argument may be
         ``None`` if the load does not correspond to a :class:`_query.Query`,
         such as during :meth:`.Session.merge`.

        .. seealso::

            :ref:`mapped_class_load_events`

            :meth:`.InstanceEvents.init`

            :meth:`.InstanceEvents.refresh`

            :meth:`.SessionEvents.loaded_as_persistent`

        """

    def refresh(self, target: _O, context: QueryContext, attrs: Optional[Iterable[str]]) -> None:
        """Receive an object instance after one or more attributes have
        been refreshed from a query.

        Contrast this to the :meth:`.InstanceEvents.load` method, which
        is invoked when the object is first loaded from a query.

        .. note:: This event is invoked within the loader process before
           eager loaders may have been completed, and the object's state may
           not be complete.  Additionally, invoking row-level refresh
           operations on the object will place the object into a new loader
           context, interfering with the existing load context.   See the note
           on :meth:`.InstanceEvents.load` for background on making use of the
           :paramref:`.InstanceEvents.restore_load_context` parameter, in
           order to resolve this scenario.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param context: the :class:`.QueryContext` corresponding to the
         current :class:`_query.Query` in progress.
        :param attrs: sequence of attribute names which
         were populated, or None if all column-mapped, non-deferred
         attributes were populated.

        .. seealso::

            :ref:`mapped_class_load_events`

            :meth:`.InstanceEvents.load`

        """

    def refresh_flush(self, target: _O, flush_context: UOWTransaction, attrs: Optional[Iterable[str]]) -> None:
        """Receive an object instance after one or more attributes that
        contain a column-level default or onupdate handler have been refreshed
        during persistence of the object's state.

        This event is the same as :meth:`.InstanceEvents.refresh` except
        it is invoked within the unit of work flush process, and includes
        only non-primary-key columns that have column level default or
        onupdate handlers, including Python callables as well as server side
        defaults and triggers which may be fetched via the RETURNING clause.

        .. note::

            While the :meth:`.InstanceEvents.refresh_flush` event is triggered
            for an object that was INSERTed as well as for an object that was
            UPDATEd, the event is geared primarily  towards the UPDATE process;
            it is mostly an internal artifact that INSERT actions can also
            trigger this event, and note that **primary key columns for an
            INSERTed row are explicitly omitted** from this event.  In order to
            intercept the newly INSERTed state of an object, the
            :meth:`.SessionEvents.pending_to_persistent` and
            :meth:`.MapperEvents.after_insert` are better choices.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param flush_context: Internal :class:`.UOWTransaction` object
         which handles the details of the flush.
        :param attrs: sequence of attribute names which
         were populated.

        .. seealso::

            :ref:`mapped_class_load_events`

            :ref:`orm_server_defaults`

            :ref:`metadata_defaults_toplevel`

        """

    def expire(self, target: _O, attrs: Optional[Iterable[str]]) -> None:
        """Receive an object instance after its attributes or some subset
        have been expired.

        'keys' is a list of attribute names.  If None, the entire
        state was expired.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param attrs: sequence of attribute
         names which were expired, or None if all attributes were
         expired.

        """

    def pickle(self, target: _O, state_dict: _InstanceDict) -> None:
        """Receive an object instance when its associated state is
        being pickled.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param state_dict: the dictionary returned by
         :class:`.InstanceState.__getstate__`, containing the state
         to be pickled.

        """

    def unpickle(self, target: _O, state_dict: _InstanceDict) -> None:
        """Receive an object instance after its associated state has
        been unpickled.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param state_dict: the dictionary sent to
         :class:`.InstanceState.__setstate__`, containing the state
         dictionary which was pickled.

        """