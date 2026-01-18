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
class AttributeImpl:
    """internal implementation for instrumented attributes."""
    collection: bool
    default_accepts_scalar_loader: bool
    uses_objects: bool
    supports_population: bool
    dynamic: bool
    _is_has_collection_adapter = False
    _replace_token: AttributeEventToken
    _remove_token: AttributeEventToken
    _append_token: AttributeEventToken

    def __init__(self, class_: _ExternalEntityType[_O], key: str, callable_: Optional[_LoaderCallable], dispatch: _Dispatch[QueryableAttribute[Any]], trackparent: bool=False, compare_function: Optional[Callable[..., bool]]=None, active_history: bool=False, parent_token: Optional[AttributeEventToken]=None, load_on_unexpire: bool=True, send_modified_events: bool=True, accepts_scalar_loader: Optional[bool]=None, **kwargs: Any):
        """Construct an AttributeImpl.

        :param \\class_: associated class

        :param key: string name of the attribute

        :param \\callable_:
          optional function which generates a callable based on a parent
          instance, which produces the "default" values for a scalar or
          collection attribute when it's first accessed, if not present
          already.

        :param trackparent:
          if True, attempt to track if an instance has a parent attached
          to it via this attribute.

        :param compare_function:
          a function that compares two values which are normally
          assignable to this attribute.

        :param active_history:
          indicates that get_history() should always return the "old" value,
          even if it means executing a lazy callable upon attribute change.

        :param parent_token:
          Usually references the MapperProperty, used as a key for
          the hasparent() function to identify an "owning" attribute.
          Allows multiple AttributeImpls to all match a single
          owner attribute.

        :param load_on_unexpire:
          if False, don't include this attribute in a load-on-expired
          operation, i.e. the "expired_attribute_loader" process.
          The attribute can still be in the "expired" list and be
          considered to be "expired".   Previously, this flag was called
          "expire_missing" and is only used by a deferred column
          attribute.

        :param send_modified_events:
          if False, the InstanceState._modified_event method will have no
          effect; this means the attribute will never show up as changed in a
          history entry.

        """
        self.class_ = class_
        self.key = key
        self.callable_ = callable_
        self.dispatch = dispatch
        self.trackparent = trackparent
        self.parent_token = parent_token or self
        self.send_modified_events = send_modified_events
        if compare_function is None:
            self.is_equal = operator.eq
        else:
            self.is_equal = compare_function
        if accepts_scalar_loader is not None:
            self.accepts_scalar_loader = accepts_scalar_loader
        else:
            self.accepts_scalar_loader = self.default_accepts_scalar_loader
        _deferred_history = kwargs.pop('_deferred_history', False)
        self._deferred_history = _deferred_history
        if active_history:
            self.dispatch._active_history = True
        self.load_on_unexpire = load_on_unexpire
        self._modified_token = AttributeEventToken(self, OP_MODIFIED)
    __slots__ = ('class_', 'key', 'callable_', 'dispatch', 'trackparent', 'parent_token', 'send_modified_events', 'is_equal', 'load_on_unexpire', '_modified_token', 'accepts_scalar_loader', '_deferred_history')

    def __str__(self) -> str:
        return f'{self.class_.__name__}.{self.key}'

    def _get_active_history(self):
        """Backwards compat for impl.active_history"""
        return self.dispatch._active_history

    def _set_active_history(self, value):
        self.dispatch._active_history = value
    active_history = property(_get_active_history, _set_active_history)

    def hasparent(self, state: InstanceState[Any], optimistic: bool=False) -> bool:
        """Return the boolean value of a `hasparent` flag attached to
        the given state.

        The `optimistic` flag determines what the default return value
        should be if no `hasparent` flag can be located.

        As this function is used to determine if an instance is an
        *orphan*, instances that were loaded from storage should be
        assumed to not be orphans, until a True/False value for this
        flag is set.

        An instance attribute that is loaded by a callable function
        will also not have a `hasparent` flag.

        """
        msg = 'This AttributeImpl is not configured to track parents.'
        assert self.trackparent, msg
        return state.parents.get(id(self.parent_token), optimistic) is not False

    def sethasparent(self, state: InstanceState[Any], parent_state: InstanceState[Any], value: bool) -> None:
        """Set a boolean flag on the given item corresponding to
        whether or not it is attached to a parent object via the
        attribute represented by this ``InstrumentedAttribute``.

        """
        msg = 'This AttributeImpl is not configured to track parents.'
        assert self.trackparent, msg
        id_ = id(self.parent_token)
        if value:
            state.parents[id_] = parent_state
        else:
            if id_ in state.parents:
                last_parent = state.parents[id_]
                if last_parent is not False and last_parent.key != parent_state.key:
                    if last_parent.obj() is None:
                        raise orm_exc.StaleDataError("Removing state %s from parent state %s along attribute '%s', but the parent record has gone stale, can't be sure this is the most recent parent." % (state_str(state), state_str(parent_state), self.key))
                    return
            state.parents[id_] = False

    def get_history(self, state: InstanceState[Any], dict_: _InstanceDict, passive: PassiveFlag=PASSIVE_OFF) -> History:
        raise NotImplementedError()

    def get_all_pending(self, state: InstanceState[Any], dict_: _InstanceDict, passive: PassiveFlag=PASSIVE_NO_INITIALIZE) -> _AllPendingType:
        """Return a list of tuples of (state, obj)
        for all objects in this attribute's current state
        + history.

        Only applies to object-based attributes.

        This is an inlining of existing functionality
        which roughly corresponds to:

            get_state_history(
                        state,
                        key,
                        passive=PASSIVE_NO_INITIALIZE).sum()

        """
        raise NotImplementedError()

    def _default_value(self, state: InstanceState[Any], dict_: _InstanceDict) -> Any:
        """Produce an empty value for an uninitialized scalar attribute."""
        assert self.key not in dict_, '_default_value should only be invoked for an uninitialized or expired attribute'
        value = None
        for fn in self.dispatch.init_scalar:
            ret = fn(state, value, dict_)
            if ret is not ATTR_EMPTY:
                value = ret
        return value

    def get(self, state: InstanceState[Any], dict_: _InstanceDict, passive: PassiveFlag=PASSIVE_OFF) -> Any:
        """Retrieve a value from the given object.
        If a callable is assembled on this object's attribute, and
        passive is False, the callable will be executed and the
        resulting value will be set as the new value for this attribute.
        """
        if self.key in dict_:
            return dict_[self.key]
        else:
            key = self.key
            if key not in state.committed_state or state.committed_state[key] is NO_VALUE:
                if not passive & CALLABLES_OK:
                    return PASSIVE_NO_RESULT
                value = self._fire_loader_callables(state, key, passive)
                if value is PASSIVE_NO_RESULT or value is NO_VALUE:
                    return value
                elif value is ATTR_WAS_SET:
                    try:
                        return dict_[key]
                    except KeyError as err:
                        raise KeyError('Deferred loader for attribute %r failed to populate correctly' % key) from err
                elif value is not ATTR_EMPTY:
                    return self.set_committed_value(state, dict_, value)
            if not passive & INIT_OK:
                return NO_VALUE
            else:
                return self._default_value(state, dict_)

    def _fire_loader_callables(self, state: InstanceState[Any], key: str, passive: PassiveFlag) -> Any:
        if self.accepts_scalar_loader and self.load_on_unexpire and (key in state.expired_attributes):
            return state._load_expired(state, passive)
        elif key in state.callables:
            callable_ = state.callables[key]
            return callable_(state, passive)
        elif self.callable_:
            return self.callable_(state, passive)
        else:
            return ATTR_EMPTY

    def append(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken], passive: PassiveFlag=PASSIVE_OFF) -> None:
        self.set(state, dict_, value, initiator, passive=passive)

    def remove(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken], passive: PassiveFlag=PASSIVE_OFF) -> None:
        self.set(state, dict_, None, initiator, passive=passive, check_old=value)

    def pop(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken], passive: PassiveFlag=PASSIVE_OFF) -> None:
        self.set(state, dict_, None, initiator, passive=passive, check_old=value, pop=True)

    def set(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken]=None, passive: PassiveFlag=PASSIVE_OFF, check_old: Any=None, pop: bool=False) -> None:
        raise NotImplementedError()

    def delete(self, state: InstanceState[Any], dict_: _InstanceDict) -> None:
        raise NotImplementedError()

    def get_committed_value(self, state: InstanceState[Any], dict_: _InstanceDict, passive: PassiveFlag=PASSIVE_OFF) -> Any:
        """return the unchanged value of this attribute"""
        if self.key in state.committed_state:
            value = state.committed_state[self.key]
            if value is NO_VALUE:
                return None
            else:
                return value
        else:
            return self.get(state, dict_, passive=passive)

    def set_committed_value(self, state, dict_, value):
        """set an attribute value on the given instance and 'commit' it."""
        dict_[self.key] = value
        state._commit(dict_, [self.key])
        return value