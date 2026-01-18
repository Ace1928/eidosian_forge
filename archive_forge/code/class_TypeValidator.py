from abc import ABC, abstractmethod
from typing import TypeVar, Union
class TypeValidator(Validator):

    def __init__(self, attr_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_type = attr_type
        try:
            origin = attr_type.__origin__
            subtypes = attr_type.__args__
        except AttributeError:
            self.attr_type = (attr_type,)
        else:
            if origin is Union:
                self.attr_type = subtypes
            else:
                raise TypeError(f'{attr_type} is not currently supported.')

    def call(self, attr_name, value):
        if not isinstance(value, self.attr_type):
            raise TypeError(f'{attr_name} must be of type {self.attr_type!r} (got {type(value)!r})')