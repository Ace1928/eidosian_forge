import sys
import types
import typing
import warnings
import simdjson as json
from enum import Enum
from dataclasses import is_dataclass
from .utils import issubclass_safe
def _process_class_internal(cls):

    def _temp_from_dict(cls, *args, **kwarg):
        _replace_from_dict(cls, '_lazyclasses_from_dict')
        return cls._lazyclasses_from_dict(*args, **kwarg)
    cls.from_dict = classmethod(_temp_from_dict)
    cls._lazyclasses_from_dict = classmethod(_temp_from_dict)

    def _temp_to_dict(self, *args, **kwargs):
        _replace_to_dict(cls, '_lazyclasses_to_dict')
        return self._lazyclasses_to_dict(*args, **kwargs)
    cls.to_dict = _temp_to_dict
    cls._lazyclasses_to_dict = _temp_to_dict
    return cls