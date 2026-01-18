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
class CollectionAttributeImpl(HasCollectionAdapter, AttributeImpl):
    """A collection-holding attribute that instruments changes in membership.

    Only handles collections of instrumented objects.

    InstrumentedCollectionAttribute holds an arbitrary, user-specified
    container object (defaulting to a list) and brokers access to the
    CollectionAdapter, a "view" onto that object that presents consistent bag
    semantics to the orm layer independent of the user data implementation.

    """
    uses_objects = True
    collection = True
    default_accepts_scalar_loader = False
    supports_population = True
    dynamic = False
    _bulk_replace_token: AttributeEventToken
    __slots__ = ('copy', 'collection_factory', '_append_token', '_remove_token', '_bulk_replace_token', '_duck_typed_as')

    def __init__(self, class_, key, callable_, dispatch, typecallable=None, trackparent=False, copy_function=None, compare_function=None, **kwargs):
        super().__init__(class_, key, callable_, dispatch, trackparent=trackparent, compare_function=compare_function, **kwargs)
        if copy_function is None:
            copy_function = self.__copy
        self.copy = copy_function
        self.collection_factory = typecallable
        self._append_token = AttributeEventToken(self, OP_APPEND)
        self._remove_token = AttributeEventToken(self, OP_REMOVE)
        self._bulk_replace_token = AttributeEventToken(self, OP_BULK_REPLACE)
        self._duck_typed_as = util.duck_type_collection(self.collection_factory())
        if getattr(self.collection_factory, '_sa_linker', None):

            @event.listens_for(self, 'init_collection')
            def link(target, collection, collection_adapter):
                collection._sa_linker(collection_adapter)

            @event.listens_for(self, 'dispose_collection')
            def unlink(target, collection, collection_adapter):
                collection._sa_linker(None)

    def __copy(self, item):
        return [y for y in collections.collection_adapter(item)]

    def get_history(self, state: InstanceState[Any], dict_: _InstanceDict, passive: PassiveFlag=PASSIVE_OFF) -> History:
        current = self.get(state, dict_, passive=passive)
        if current is PASSIVE_NO_RESULT:
            if passive & PassiveFlag.INCLUDE_PENDING_MUTATIONS and self.key in state._pending_mutations:
                pending = state._pending_mutations[self.key]
                return pending.merge_with_history(HISTORY_BLANK)
            else:
                return HISTORY_BLANK
        else:
            if passive & PassiveFlag.INCLUDE_PENDING_MUTATIONS:
                assert self.key not in state._pending_mutations
            return History.from_collection(self, state, current)

    def get_all_pending(self, state: InstanceState[Any], dict_: _InstanceDict, passive: PassiveFlag=PASSIVE_NO_INITIALIZE) -> _AllPendingType:
        if self.key not in dict_:
            return []
        current = dict_[self.key]
        current = getattr(current, '_sa_adapter')
        if self.key in state.committed_state:
            original = state.committed_state[self.key]
            if original is not NO_VALUE:
                current_states = [(c is not None and instance_state(c) or None, c) for c in current]
                original_states = [(c is not None and instance_state(c) or None, c) for c in original]
                current_set = dict(current_states)
                original_set = dict(original_states)
                return [(s, o) for s, o in current_states if s not in original_set] + [(s, o) for s, o in current_states if s in original_set] + [(s, o) for s, o in original_states if s not in current_set]
        return [(instance_state(o), o) for o in current]

    def fire_append_event(self, state: InstanceState[Any], dict_: _InstanceDict, value: _T, initiator: Optional[AttributeEventToken], key: Optional[Any]) -> _T:
        for fn in self.dispatch.append:
            value = fn(state, value, initiator or self._append_token, key=key)
        state._modified_event(dict_, self, NO_VALUE, True)
        if self.trackparent and value is not None:
            self.sethasparent(instance_state(value), state, True)
        return value

    def fire_append_wo_mutation_event(self, state: InstanceState[Any], dict_: _InstanceDict, value: _T, initiator: Optional[AttributeEventToken], key: Optional[Any]) -> _T:
        for fn in self.dispatch.append_wo_mutation:
            value = fn(state, value, initiator or self._append_token, key=key)
        return value

    def fire_pre_remove_event(self, state: InstanceState[Any], dict_: _InstanceDict, initiator: Optional[AttributeEventToken], key: Optional[Any]) -> None:
        """A special event used for pop() operations.

        The "remove" event needs to have the item to be removed passed to
        it, which in the case of pop from a set, we don't have a way to access
        the item before the operation.   the event is used for all pop()
        operations (even though set.pop is the one where it is really needed).

        """
        state._modified_event(dict_, self, NO_VALUE, True)

    def fire_remove_event(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken], key: Optional[Any]) -> None:
        if self.trackparent and value is not None:
            self.sethasparent(instance_state(value), state, False)
        for fn in self.dispatch.remove:
            fn(state, value, initiator or self._remove_token, key=key)
        state._modified_event(dict_, self, NO_VALUE, True)

    def delete(self, state: InstanceState[Any], dict_: _InstanceDict) -> None:
        if self.key not in dict_:
            return
        state._modified_event(dict_, self, NO_VALUE, True)
        collection = self.get_collection(state, state.dict)
        collection.clear_with_event()
        del dict_[self.key]

    def _default_value(self, state: InstanceState[Any], dict_: _InstanceDict) -> _AdaptedCollectionProtocol:
        """Produce an empty collection for an un-initialized attribute"""
        assert self.key not in dict_, '_default_value should only be invoked for an uninitialized or expired attribute'
        if self.key in state._empty_collections:
            return state._empty_collections[self.key]
        adapter, user_data = self._initialize_collection(state)
        adapter._set_empty(user_data)
        return user_data

    def _initialize_collection(self, state: InstanceState[Any]) -> Tuple[CollectionAdapter, _AdaptedCollectionProtocol]:
        adapter, collection = state.manager.initialize_collection(self.key, state, self.collection_factory)
        self.dispatch.init_collection(state, collection, adapter)
        return (adapter, collection)

    def append(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken], passive: PassiveFlag=PASSIVE_OFF) -> None:
        collection = self.get_collection(state, dict_, user_data=None, passive=passive)
        if collection is PASSIVE_NO_RESULT:
            value = self.fire_append_event(state, dict_, value, initiator, key=NO_KEY)
            assert self.key not in dict_, 'Collection was loaded during event handling.'
            state._get_pending_mutation(self.key).append(value)
        else:
            if TYPE_CHECKING:
                assert isinstance(collection, CollectionAdapter)
            collection.append_with_event(value, initiator)

    def remove(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken], passive: PassiveFlag=PASSIVE_OFF) -> None:
        collection = self.get_collection(state, state.dict, user_data=None, passive=passive)
        if collection is PASSIVE_NO_RESULT:
            self.fire_remove_event(state, dict_, value, initiator, key=NO_KEY)
            assert self.key not in dict_, 'Collection was loaded during event handling.'
            state._get_pending_mutation(self.key).remove(value)
        else:
            if TYPE_CHECKING:
                assert isinstance(collection, CollectionAdapter)
            collection.remove_with_event(value, initiator)

    def pop(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken], passive: PassiveFlag=PASSIVE_OFF) -> None:
        try:
            self.remove(state, dict_, value, initiator, passive=passive)
        except (ValueError, KeyError, IndexError):
            pass

    def set(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any, initiator: Optional[AttributeEventToken]=None, passive: PassiveFlag=PassiveFlag.PASSIVE_OFF, check_old: Any=None, pop: bool=False, _adapt: bool=True) -> None:
        iterable = orig_iterable = value
        new_keys = None
        new_collection, user_data = self._initialize_collection(state)
        if _adapt:
            if new_collection._converter is not None:
                iterable = new_collection._converter(iterable)
            else:
                setting_type = util.duck_type_collection(iterable)
                receiving_type = self._duck_typed_as
                if setting_type is not receiving_type:
                    given = iterable is None and 'None' or iterable.__class__.__name__
                    wanted = self._duck_typed_as.__name__
                    raise TypeError('Incompatible collection type: %s is not %s-like' % (given, wanted))
                if hasattr(iterable, '_sa_iterator'):
                    iterable = iterable._sa_iterator()
                elif setting_type is dict:
                    new_keys = list(iterable)
                    iterable = iterable.values()
                else:
                    iterable = iter(iterable)
        elif util.duck_type_collection(iterable) is dict:
            new_keys = list(value)
        new_values = list(iterable)
        evt = self._bulk_replace_token
        self.dispatch.bulk_replace(state, new_values, evt, keys=new_keys)
        old = self.get(state, dict_, passive=PASSIVE_ONLY_PERSISTENT ^ passive & PassiveFlag.NO_RAISE)
        if old is PASSIVE_NO_RESULT:
            old = self._default_value(state, dict_)
        elif old is orig_iterable:
            return
        state._modified_event(dict_, self, old, True)
        old_collection = old._sa_adapter
        dict_[self.key] = user_data
        collections.bulk_replace(new_values, old_collection, new_collection, initiator=evt)
        self._dispose_previous_collection(state, old, old_collection, True)

    def _dispose_previous_collection(self, state: InstanceState[Any], collection: _AdaptedCollectionProtocol, adapter: CollectionAdapter, fire_event: bool) -> None:
        del collection._sa_adapter
        state._empty_collections.pop(self.key, None)
        if fire_event:
            self.dispatch.dispose_collection(state, collection, adapter)

    def _invalidate_collection(self, collection: _AdaptedCollectionProtocol) -> None:
        adapter = getattr(collection, '_sa_adapter')
        adapter.invalidated = True

    def set_committed_value(self, state: InstanceState[Any], dict_: _InstanceDict, value: Any) -> _AdaptedCollectionProtocol:
        """Set an attribute value on the given instance and 'commit' it."""
        collection, user_data = self._initialize_collection(state)
        if value:
            collection.append_multiple_without_event(value)
        state.dict[self.key] = user_data
        state._commit(dict_, [self.key])
        if self.key in state._pending_mutations:
            state._modified_event(dict_, self, user_data, True)
            pending = state._pending_mutations.pop(self.key)
            added = pending.added_items
            removed = pending.deleted_items
            for item in added:
                collection.append_without_event(item)
            for item in removed:
                collection.remove_without_event(item)
        return user_data

    @overload
    def get_collection(self, state: InstanceState[Any], dict_: _InstanceDict, user_data: Literal[None]=..., passive: Literal[PassiveFlag.PASSIVE_OFF]=...) -> CollectionAdapter:
        ...

    @overload
    def get_collection(self, state: InstanceState[Any], dict_: _InstanceDict, user_data: _AdaptedCollectionProtocol=..., passive: PassiveFlag=...) -> CollectionAdapter:
        ...

    @overload
    def get_collection(self, state: InstanceState[Any], dict_: _InstanceDict, user_data: Optional[_AdaptedCollectionProtocol]=..., passive: PassiveFlag=PASSIVE_OFF) -> Union[Literal[LoaderCallableStatus.PASSIVE_NO_RESULT], CollectionAdapter]:
        ...

    def get_collection(self, state: InstanceState[Any], dict_: _InstanceDict, user_data: Optional[_AdaptedCollectionProtocol]=None, passive: PassiveFlag=PASSIVE_OFF) -> Union[Literal[LoaderCallableStatus.PASSIVE_NO_RESULT], CollectionAdapter]:
        """Retrieve the CollectionAdapter associated with the given state.

        if user_data is None, retrieves it from the state using normal
        "get()" rules, which will fire lazy callables or return the "empty"
        collection value.

        """
        if user_data is None:
            fetch_user_data = self.get(state, dict_, passive=passive)
            if fetch_user_data is LoaderCallableStatus.PASSIVE_NO_RESULT:
                return fetch_user_data
            else:
                user_data = cast('_AdaptedCollectionProtocol', fetch_user_data)
        return user_data._sa_adapter