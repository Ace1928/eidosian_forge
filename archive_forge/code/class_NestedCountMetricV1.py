from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Tuple, Dict, Iterator, Any, Type, Set, Iterable, TYPE_CHECKING
from .debug import get_autologger
class NestedCountMetricV1(BaseModel):
    """
    Nested Count Metric Container

    {
        'website.com': {
            '2022-01-01': 1,
            '2022-01-02': 2,
        },
        'www.website.com': {
            '2022-01-01': 3,
            '2022-01-02': 4,
        },
    }
    """
    name: Optional[str] = 'nested_count'
    data: Dict[str, CountMetric] = Field(default_factory=dict, description='The nested count values')

    def items(self, sort: Optional[bool]=None):
        """
        Returns the dict_items view of the data
        """
        return {k: dict(v.items(sort=sort)) for k, v in self.data.items()}.items()

    def __getitem__(self, key: str) -> CountMetric:
        """
        Gets the value for the given key
        """
        if key not in self.data:
            self.data[key] = CountMetric(name=key)
        return self.data[key]

    def __setitem__(self, key: str, value: CountMetric):
        """
        Sets the value for the given key
        """
        self.data[key] = value

    def __getattr__(self, name: str) -> CountMetric:
        """
        Gets the value for the given key
        """
        if name not in self.data:
            self.data[name] = CountMetric(name=name)
        return self.data[name]

    def __setattr__(self, name: str, value: CountMetric) -> None:
        """
        Sets the value for the given key
        """
        self.data[name] = value

    def __repr__(self) -> str:
        """
        Representation of the object
        """
        return f'{dict(self.items())}'

    def __str__(self) -> str:
        """
        Representation of the object
        """
        return self.__repr__()