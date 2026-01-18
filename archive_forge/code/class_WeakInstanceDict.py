from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
import weakref
from . import util as orm_util
from .. import exc as sa_exc
class WeakInstanceDict(IdentityMap):
    _dict: Dict[_IdentityKeyType[Any], InstanceState[Any]]

    def __getitem__(self, key: _IdentityKeyType[_O]) -> _O:
        state = cast('InstanceState[_O]', self._dict[key])
        o = state.obj()
        if o is None:
            raise KeyError(key)
        return o

    def __contains__(self, key: _IdentityKeyType[Any]) -> bool:
        try:
            if key in self._dict:
                state = self._dict[key]
                o = state.obj()
            else:
                return False
        except KeyError:
            return False
        else:
            return o is not None

    def contains_state(self, state: InstanceState[Any]) -> bool:
        if state.key in self._dict:
            if TYPE_CHECKING:
                assert state.key is not None
            try:
                return self._dict[state.key] is state
            except KeyError:
                return False
        else:
            return False

    def replace(self, state: InstanceState[Any]) -> Optional[InstanceState[Any]]:
        assert state.key is not None
        if state.key in self._dict:
            try:
                existing = existing_non_none = self._dict[state.key]
            except KeyError:
                existing = None
            else:
                if existing_non_none is not state:
                    self._manage_removed_state(existing_non_none)
                else:
                    return None
        else:
            existing = None
        self._dict[state.key] = state
        self._manage_incoming_state(state)
        return existing

    def add(self, state: InstanceState[Any]) -> bool:
        key = state.key
        assert key is not None
        if key in self._dict:
            try:
                existing_state = self._dict[key]
            except KeyError:
                pass
            else:
                if existing_state is not state:
                    o = existing_state.obj()
                    if o is not None:
                        raise sa_exc.InvalidRequestError("Can't attach instance %s; another instance with key %s is already present in this session." % (orm_util.state_str(state), state.key))
                else:
                    return False
        self._dict[key] = state
        self._manage_incoming_state(state)
        return True

    def _add_unpresent(self, state: InstanceState[Any], key: _IdentityKeyType[Any]) -> None:
        self._dict[key] = state
        state._instance_dict = self._wr

    def fast_get_state(self, key: _IdentityKeyType[_O]) -> Optional[InstanceState[_O]]:
        return self._dict.get(key)

    def get(self, key: _IdentityKeyType[_O], default: Optional[_O]=None) -> Optional[_O]:
        if key not in self._dict:
            return default
        try:
            state = cast('InstanceState[_O]', self._dict[key])
        except KeyError:
            return default
        else:
            o = state.obj()
            if o is None:
                return default
            return o

    def items(self) -> List[Tuple[_IdentityKeyType[Any], InstanceState[Any]]]:
        values = self.all_states()
        result = []
        for state in values:
            value = state.obj()
            key = state.key
            assert key is not None
            if value is not None:
                result.append((key, value))
        return result

    def values(self) -> List[object]:
        values = self.all_states()
        result = []
        for state in values:
            value = state.obj()
            if value is not None:
                result.append(value)
        return result

    def __iter__(self) -> Iterator[_IdentityKeyType[Any]]:
        return iter(self.keys())

    def all_states(self) -> List[InstanceState[Any]]:
        return list(self._dict.values())

    def _fast_discard(self, state: InstanceState[Any]) -> None:
        key = state.key
        assert key is not None
        try:
            st = self._dict[key]
        except KeyError:
            pass
        else:
            if st is state:
                self._dict.pop(key, None)

    def discard(self, state: InstanceState[Any]) -> None:
        self.safe_discard(state)

    def safe_discard(self, state: InstanceState[Any]) -> None:
        key = state.key
        if key in self._dict:
            assert key is not None
            try:
                st = self._dict[key]
            except KeyError:
                pass
            else:
                if st is state:
                    self._dict.pop(key, None)
                    self._manage_removed_state(state)