import sys
import types
import typing
import warnings
import simdjson as json
from enum import Enum
from dataclasses import is_dataclass
from .utils import issubclass_safe
def _temp_from_dict(cls, *args, **kwarg):
    _replace_from_dict(cls, '_lazyclasses_from_dict')
    return cls._lazyclasses_from_dict(*args, **kwarg)