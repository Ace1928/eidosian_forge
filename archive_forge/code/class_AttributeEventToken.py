from __future__ import annotations
import dataclasses
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import collections
from . import exc as orm_exc
from . import interfaces
from ._typing import insp_is_aliased_class
from .base import _DeclarativeMapped
from .base import ATTR_EMPTY
from .base import ATTR_WAS_SET
from .base import CALLABLES_OK
from .base import DEFERRED_HISTORY_LOAD
from .base import INCLUDE_PENDING_MUTATIONS  # noqa
from .base import INIT_OK
from .base import instance_dict as instance_dict
from .base import instance_state as instance_state
from .base import instance_str
from .base import LOAD_AGAINST_COMMITTED
from .base import LoaderCallableStatus
from .base import manager_of_class as manager_of_class
from .base import Mapped as Mapped  # noqa
from .base import NEVER_SET  # noqa
from .base import NO_AUTOFLUSH
from .base import NO_CHANGE  # noqa
from .base import NO_KEY
from .base import NO_RAISE
from .base import NO_VALUE
from .base import NON_PERSISTENT_OK  # noqa
from .base import opt_manager_of_class as opt_manager_of_class
from .base import PASSIVE_CLASS_MISMATCH  # noqa
from .base import PASSIVE_NO_FETCH
from .base import PASSIVE_NO_FETCH_RELATED  # noqa
from .base import PASSIVE_NO_INITIALIZE
from .base import PASSIVE_NO_RESULT
from .base import PASSIVE_OFF
from .base import PASSIVE_ONLY_PERSISTENT
from .base import PASSIVE_RETURN_NO_VALUE
from .base import PassiveFlag
from .base import RELATED_OBJECT_OK  # noqa
from .base import SQL_OK  # noqa
from .base import SQLORMExpression
from .base import state_str
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..event import dispatcher
from ..event import EventTarget
from ..sql import base as sql_base
from ..sql import cache_key
from ..sql import coercions
from ..sql import roles
from ..sql import visitors
from ..sql.cache_key import HasCacheKey
from ..sql.visitors import _TraverseInternalsType
from ..sql.visitors import InternalTraversal
from ..util.typing import Literal
from ..util.typing import Self
from ..util.typing import TypeGuard
class AttributeEventToken:
    """A token propagated throughout the course of a chain of attribute
    events.

    Serves as an indicator of the source of the event and also provides
    a means of controlling propagation across a chain of attribute
    operations.

    The :class:`.Event` object is sent as the ``initiator`` argument
    when dealing with events such as :meth:`.AttributeEvents.append`,
    :meth:`.AttributeEvents.set`,
    and :meth:`.AttributeEvents.remove`.

    The :class:`.Event` object is currently interpreted by the backref
    event handlers, and is used to control the propagation of operations
    across two mutually-dependent attributes.

    .. versionchanged:: 2.0  Changed the name from ``AttributeEvent``
       to ``AttributeEventToken``.

    :attribute impl: The :class:`.AttributeImpl` which is the current event
     initiator.

    :attribute op: The symbol :attr:`.OP_APPEND`, :attr:`.OP_REMOVE`,
     :attr:`.OP_REPLACE`, or :attr:`.OP_BULK_REPLACE`, indicating the
     source operation.

    """
    __slots__ = ('impl', 'op', 'parent_token')

    def __init__(self, attribute_impl: AttributeImpl, op: util.symbol):
        self.impl = attribute_impl
        self.op = op
        self.parent_token = self.impl.parent_token

    def __eq__(self, other):
        return isinstance(other, AttributeEventToken) and other.impl is self.impl and (other.op == self.op)

    @property
    def key(self):
        return self.impl.key

    def hasparent(self, state):
        return self.impl.hasparent(state)