import collections.abc
import copy
import pickle
from typing import (
class ScalarMap(MutableMapping[_K, _V]):
    """Simple, type-checked, dict-like container for holding repeated scalars."""
    __slots__ = ['_key_checker', '_value_checker', '_values', '_message_listener', '_entry_descriptor']

    def __init__(self, message_listener: Any, key_checker: Any, value_checker: Any, entry_descriptor: Any) -> None:
        """
    Args:
      message_listener: A MessageListener implementation.
        The ScalarMap will call this object's Modified() method when it
        is modified.
      key_checker: A type_checkers.ValueChecker instance to run on keys
        inserted into this container.
      value_checker: A type_checkers.ValueChecker instance to run on values
        inserted into this container.
      entry_descriptor: The MessageDescriptor of a map entry: key and value.
    """
        self._message_listener = message_listener
        self._key_checker = key_checker
        self._value_checker = value_checker
        self._entry_descriptor = entry_descriptor
        self._values = {}

    def __getitem__(self, key: _K) -> _V:
        try:
            return self._values[key]
        except KeyError:
            key = self._key_checker.CheckValue(key)
            val = self._value_checker.DefaultValue()
            self._values[key] = val
            return val

    def __contains__(self, item: _K) -> bool:
        self._key_checker.CheckValue(item)
        return item in self._values

    @overload
    def get(self, key: _K) -> Optional[_V]:
        ...

    @overload
    def get(self, key: _K, default: _T) -> Union[_V, _T]:
        ...

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def __setitem__(self, key: _K, value: _V) -> _T:
        checked_key = self._key_checker.CheckValue(key)
        checked_value = self._value_checker.CheckValue(value)
        self._values[checked_key] = checked_value
        self._message_listener.Modified()

    def __delitem__(self, key: _K) -> None:
        del self._values[key]
        self._message_listener.Modified()

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[_K]:
        return iter(self._values)

    def __repr__(self) -> str:
        return repr(self._values)

    def MergeFrom(self, other: 'ScalarMap[_K, _V]') -> None:
        self._values.update(other._values)
        self._message_listener.Modified()

    def InvalidateIterators(self) -> None:
        original = self._values
        self._values = original.copy()
        original[None] = None

    def clear(self) -> None:
        self._values.clear()
        self._message_listener.Modified()

    def GetEntryClass(self) -> Any:
        return self._entry_descriptor._concrete_class