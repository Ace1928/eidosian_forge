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
class InstrumentationEvents(event.Events[InstrumentationFactory]):
    """Events related to class instrumentation events.

    The listeners here support being established against
    any new style class, that is any object that is a subclass
    of 'type'.  Events will then be fired off for events
    against that class.  If the "propagate=True" flag is passed
    to event.listen(), the event will fire off for subclasses
    of that class as well.

    The Python ``type`` builtin is also accepted as a target,
    which when used has the effect of events being emitted
    for all classes.

    Note the "propagate" flag here is defaulted to ``True``,
    unlike the other class level events where it defaults
    to ``False``.  This means that new subclasses will also
    be the subject of these events, when a listener
    is established on a superclass.

    """
    _target_class_doc = 'SomeBaseClass'
    _dispatch_target = InstrumentationFactory

    @classmethod
    def _accept_with(cls, target: Union[InstrumentationFactory, Type[InstrumentationFactory]], identifier: str) -> Optional[Union[InstrumentationFactory, Type[InstrumentationFactory]]]:
        if isinstance(target, type):
            return _InstrumentationEventsHold(target)
        else:
            return None

    @classmethod
    def _listen(cls, event_key: _EventKey[_T], propagate: bool=True, **kw: Any) -> None:
        target, identifier, fn = (event_key.dispatch_target, event_key.identifier, event_key._listen_fn)

        def listen(target_cls: type, *arg: Any) -> Optional[Any]:
            listen_cls = target()
            if listen_cls is None:
                return None
            if propagate and issubclass(target_cls, listen_cls):
                return fn(target_cls, *arg)
            elif not propagate and target_cls is listen_cls:
                return fn(target_cls, *arg)
            else:
                return None

        def remove(ref: ReferenceType[_T]) -> None:
            key = event.registry._EventKey(None, identifier, listen, instrumentation._instrumentation_factory)
            getattr(instrumentation._instrumentation_factory.dispatch, identifier).remove(key)
        target = weakref.ref(target.class_, remove)
        event_key.with_dispatch_target(instrumentation._instrumentation_factory).with_wrapper(listen).base_listen(**kw)

    @classmethod
    def _clear(cls) -> None:
        super()._clear()
        instrumentation._instrumentation_factory.dispatch._clear()

    def class_instrument(self, cls: ClassManager[_O]) -> None:
        """Called after the given class is instrumented.

        To get at the :class:`.ClassManager`, use
        :func:`.manager_of_class`.

        """

    def class_uninstrument(self, cls: ClassManager[_O]) -> None:
        """Called before the given class is uninstrumented.

        To get at the :class:`.ClassManager`, use
        :func:`.manager_of_class`.

        """

    def attribute_instrument(self, cls: ClassManager[_O], key: _KT, inst: _O) -> None:
        """Called when an attribute is instrumented."""