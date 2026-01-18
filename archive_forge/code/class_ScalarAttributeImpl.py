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
class ScalarAttributeImpl(AttributeImpl):
    """represents a scalar value-holding InstrumentedAttribute."""
    default_accepts_scalar_loader = True
    uses_objects = False
    supports_population = True
    collection = False
    dynamic = False
    __slots__ = ('_replace_token', '_append_token', '_remove_token')

    def __init__(self, *arg, **kw):
        super().__init__(*arg, **kw)
        self._replace_token = self._append_token = AttributeEventToken(self, OP_REPLACE)
        self._remove_token = AttributeEventToken(self, OP_REMOVE)

    def delete(self, state: InstanceState[Any], dict_: _InstanceDict) -> None:
        if self.dispatch._active_history:
            old = self.get(state, dict_, PASSIVE_RETURN_NO_VALUE)
        else:
            old = dict_.get(self.key, NO_VALUE)
        if self.dispatch.remove:
            self.fire_remove_event(state, dict_, old, self._remove_token)
        state._modified_event(dict_, self, old)
        existing = dict_.pop(self.key, NO_VALUE)
        if existing is NO_VALUE and old is NO_VALUE and (not state.expired) and (self.key not in state.expired_attributes):
            raise AttributeError('%s object does not have a value' % self)

    def get_history(self, state: InstanceState[Any], dict_: Dict[str, Any], passive: PassiveFlag=PASSIVE_OFF) -> History:
        if self.key in dict_:
            return History.from_scalar_attribute(self, state, dict_[self.key])
        elif self.key in state.committed_state:
            return History.from_scalar_attribute(self, state, NO_VALUE)
        else:
            if passive & INIT_OK:
                passive ^= INIT_OK
            current = self.get(state, dict_, passive=passive)
            if current is PASSIVE_NO_RESULT:
                return HISTORY_BLANK
            else:
                return History.from_scalar_attribute(self, state, current)

    def set(self, state: InstanceState[Any], dict_: Dict[str, Any], value: Any, initiator: Optional[AttributeEventToken]=None, passive: PassiveFlag=PASSIVE_OFF, check_old: Optional[object]=None, pop: bool=False) -> None:
        if self.dispatch._active_history:
            old = self.get(state, dict_, PASSIVE_RETURN_NO_VALUE)
        else:
            old = dict_.get(self.key, NO_VALUE)
        if self.dispatch.set:
            value = self.fire_replace_event(state, dict_, value, old, initiator)
        state._modified_event(dict_, self, old)
        dict_[self.key] = value

    def fire_replace_event(self, state: InstanceState[Any], dict_: _InstanceDict, value: _T, previous: Any, initiator: Optional[AttributeEventToken]) -> _T:
        for fn in self.dispatch.set:
            value = fn(state, value, previous, initiator or self._replace_token)
        return value

    def fire_remove_event(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken]) -> None:
        for fn in self.dispatch.remove:
            fn(state, value, initiator or self._remove_token)