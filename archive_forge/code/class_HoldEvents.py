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
class HoldEvents(Generic[_ET2]):
    _dispatch_target: Optional[Type[_ET2]] = None

    @classmethod
    def _listen(cls, event_key: _EventKey[_ET2], raw: bool=False, propagate: bool=False, retval: bool=False, **kw: Any) -> None:
        target = event_key.dispatch_target
        if target.class_ in target.all_holds:
            collection = target.all_holds[target.class_]
        else:
            collection = target.all_holds[target.class_] = {}
        event.registry._stored_in_collection(event_key, target)
        collection[event_key._key] = (event_key, raw, propagate, retval, kw)
        if propagate:
            stack = list(target.class_.__subclasses__())
            while stack:
                subclass = stack.pop(0)
                stack.extend(subclass.__subclasses__())
                subject = target.resolve(subclass)
                if subject is not None:
                    event_key.with_dispatch_target(subject).listen(raw=raw, propagate=False, retval=retval, **kw)