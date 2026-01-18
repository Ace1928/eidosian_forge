import collections.abc
import copy
import pickle
from typing import (
class BaseContainer(Sequence[_T]):
    """Base container class."""
    __slots__ = ['_message_listener', '_values']

    def __init__(self, message_listener: Any) -> None:
        """
    Args:
      message_listener: A MessageListener implementation.
        The RepeatedScalarFieldContainer will call this object's
        Modified() method when it is modified.
    """
        self._message_listener = message_listener
        self._values = []

    @overload
    def __getitem__(self, key: int) -> _T:
        ...

    @overload
    def __getitem__(self, key: slice) -> List[_T]:
        ...

    def __getitem__(self, key):
        """Retrieves item by the specified key."""
        return self._values[key]

    def __len__(self) -> int:
        """Returns the number of elements in the container."""
        return len(self._values)

    def __ne__(self, other: Any) -> bool:
        """Checks if another instance isn't equal to this one."""
        return not self == other
    __hash__ = None

    def __repr__(self) -> str:
        return repr(self._values)

    def sort(self, *args, **kwargs) -> None:
        if 'sort_function' in kwargs:
            kwargs['cmp'] = kwargs.pop('sort_function')
        self._values.sort(*args, **kwargs)

    def reverse(self) -> None:
        self._values.reverse()