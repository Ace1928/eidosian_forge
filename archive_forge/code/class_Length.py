from abc import ABC, abstractmethod
from typing import TypeVar, Union
class Length(Validator):

    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def call(self, attr_name, value):
        if len(value) != self.k:
            raise ValueError(f'{attr_name} must have exactly {self.k} elements (got {len(value)!r}, elems: {value!r})')