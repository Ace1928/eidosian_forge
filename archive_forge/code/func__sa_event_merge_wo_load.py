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