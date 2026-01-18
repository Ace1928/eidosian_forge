from abc import ABC, abstractmethod
from typing import TypeVar, Union
class LayoutDict(Validator):

    def call(self, attr_name, value):
        if set(value.keys()) != {'x', 'y', 'w', 'h'}:
            raise ValueError(f'{attr_name} must be a dict containing exactly the keys `x`, y`, `w`, `h`')
        for k, v in value.items():
            if not isinstance(v, int):
                raise ValueError(f'{attr_name} key `{k}` must be of type {int} (got {type(v)!r})')