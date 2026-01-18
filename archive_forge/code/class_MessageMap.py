import collections.abc
import copy
import pickle
from typing import (
class MessageMap(MutableMapping[_K, _V]):
    """Simple, type-checked, dict-like container for with submessage values."""
    __slots__ = ['_key_checker', '_values', '_message_listener', '_message_descriptor', '_entry_descriptor']

    def __init__(self, message_listener: Any, message_descriptor: Any, key_checker: Any, entry_descriptor: Any) -> None:
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
        self._message_descriptor = message_descriptor
        self._key_checker = key_checker
        self._entry_descriptor = entry_descriptor
        self._values = {}

    def __getitem__(self, key: _K) -> _V:
        key = self._key_checker.CheckValue(key)
        try:
            return self._values[key]
        except KeyError:
            new_element = self._message_descriptor._concrete_class()
            new_element._SetListener(self._message_listener)
            self._values[key] = new_element
            self._message_listener.Modified()
            return new_element

    def get_or_create(self, key: _K) -> _V:
        """get_or_create() is an alias for getitem (ie. map[key]).

    Args:
      key: The key to get or create in the map.

    This is useful in cases where you want to be explicit that the call is
    mutating the map.  This can avoid lint errors for statements like this
    that otherwise would appear to be pointless statements:

      msg.my_map[key]
    """
        return self[key]

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

    def __contains__(self, item: _K) -> bool:
        item = self._key_checker.CheckValue(item)
        return item in self._values

    def __setitem__(self, key: _K, value: _V) -> NoReturn:
        raise ValueError('May not set values directly, call my_map[key].foo = 5')

    def __delitem__(self, key: _K) -> None:
        key = self._key_checker.CheckValue(key)
        del self._values[key]
        self._message_listener.Modified()

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[_K]:
        return iter(self._values)

    def __repr__(self) -> str:
        return repr(self._values)

    def MergeFrom(self, other: 'MessageMap[_K, _V]') -> None:
        for key in other._values:
            if key in self:
                del self[key]
            self[key].CopyFrom(other[key])

    def InvalidateIterators(self) -> None:
        original = self._values
        self._values = original.copy()
        original[None] = None

    def clear(self) -> None:
        self._values.clear()
        self._message_listener.Modified()

    def GetEntryClass(self) -> Any:
        return self._entry_descriptor._concrete_class