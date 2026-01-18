from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Tuple, Dict, Iterator, Any, Type, Set, Iterable, TYPE_CHECKING
from .debug import get_autologger
class CountMetric(BaseModel):
    """
    A container for count metrics
    This is a generic metric that can be used to track individual counts of keys

    {
        'website.com': 1,
        'www.website.com': 2,
    }
    """
    name: Optional[str] = 'count'
    data: Dict[str, Union[int, float]] = Field(default_factory=dict, description='The count values')

    def incr(self, key: str, value: Union[int, float]=1) -> None:
        """
        Increments the value for the given key
        """
        if key not in self.data:
            self.data[key] = value
        else:
            self.data[key] += value

    def decr(self, key: str, value: Union[int, float]=1) -> None:
        """
        Decrements the value for the given key
        """
        if key not in self.data:
            self.data[key] = value
        else:
            self.data[key] -= value

    def reset(self, key: Optional[str]=None) -> None:
        """
        Resets the values
        """
        if key is not None:
            self.data[key] = 0
        else:
            self.data.clear()

    def add(self, key: str) -> None:
        """
        Adds the value to the count
        """
        return self.incr(key)

    def sub(self, key: str) -> None:
        """
        Subtracts the value from the count
        """
        return self.decr(key)

    def append(self, key: str) -> None:
        """
        Appends the value to the count
        """
        return self.incr(key, value=1)

    def extend(self, *keys: str) -> None:
        """
        Extends the count
        """
        for key in keys:
            self.incr(key, value=1)

    def __getitem__(self, key: str) -> Union[int, float]:
        """
        Gets the value for the given key
        """
        return self.data[key]

    def __setitem__(self, key: str, value: Union[int, float]) -> None:
        """
        Sets the value for the given key
        """
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        """
        Deletes the value for the given key
        """
        del self.data[key]

    def __contains__(self, key: str) -> bool:
        """
        Checks if the key is in the values
        """
        return key in self.data

    def __len__(self) -> int:
        """
        Gets the length of the values
        """
        return len(self.data)

    def items(self, sort: Optional[bool]=None):
        """
        Gets the items of the values
        """
        if sort:
            return sorted(self.data.items(), key=lambda x: x[1], reverse=True)
        return list(self.data.items())

    def keys(self, sort: Optional[bool]=None) -> List[str]:
        """
        Gets the keys of the values
        """
        if sort:
            return sorted(self.data.keys(), key=lambda x: x, reverse=True)
        return list(self.data.keys())

    def values(self, sort: Optional[bool]=None) -> List[Union[int, float]]:
        """
        Gets the values of the values
        """
        if sort:
            return sorted(self.data.values(), key=lambda x: x, reverse=True)
        return list(self.data.values())

    def __iter__(self) -> Iterable[str]:
        """
        Iterates over the values
        """
        return iter(self.data.keys())

    def __iadd__(self, key: Union[str, List[str]]) -> 'CountMetric':
        """
        Adds a key to the values
        """
        if not isinstance(key, list):
            key = [key]
        for k in key:
            self.incr(k)
        return self

    def __isub__(self, key: Union[str, List[str]]) -> 'CountMetric':
        """
        Subtracts a key from the values
        """
        if not isinstance(key, list):
            key = [key]
        for k in key:
            self.decr(k)
        return self

    @property
    def count(self) -> int:
        """
        Gets the count of the values
        """
        return len(self.data)

    @property
    def total(self) -> Union[int, float]:
        """
        Gets the total value
        """
        return sum(self.data.values())

    @property
    def average(self) -> Union[int, float]:
        """
        Gets the average value
        """
        return self.total / self.count

    def top_n(self, n: int, sort: Optional[bool]=None) -> List[Tuple[str, Union[int, float]]]:
        """
        Gets the top n values
        """
        if sort:
            return sorted(self.data.items(), key=lambda x: x[1], reverse=True)[:n]
        return list(self.data.items())[:n]

    def top_n_keys(self, n: int, sort: Optional[bool]=None) -> List[str]:
        """
        Gets the top n keys
        """
        if sort:
            return sorted(self.data.keys(), key=lambda x: x, reverse=True)[:n]
        return list(self.data.keys())[:n]

    def top_n_values(self, n: int, sort: Optional[bool]=None) -> List[Union[int, float]]:
        """
        Gets the top n values
        """
        if sort:
            return sorted(self.data.values(), key=lambda x: x, reverse=True)[:n]
        return list(self.data.values())[:n]

    def top_n_items(self, n: int, sort: Optional[bool]=None) -> Dict[str, Union[int, float]]:
        """
        Gets the top n items
        """
        if sort:
            return dict(sorted(self.data.items(), key=lambda x: x[1], reverse=True)[:n])
        return dict(list(self.data.items())[:n])

    def __repr__(self) -> str:
        """
        Representation of the object
        """
        return f'{dict(self.items())}'

    @property
    def key_list(self) -> List[str]:
        """
        Returns the list of keys
        """
        return list(self.data.keys())