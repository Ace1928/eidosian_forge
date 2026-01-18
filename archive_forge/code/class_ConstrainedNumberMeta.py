import abc
import math
import re
import warnings
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from types import new_class
from typing import (
from uuid import UUID
from weakref import WeakSet
from . import errors
from .datetime_parse import parse_date
from .utils import import_string, update_not_none
from .validators import (
class ConstrainedNumberMeta(type):

    def __new__(cls, name: str, bases: Any, dct: Dict[str, Any]) -> 'ConstrainedInt':
        new_cls = cast('ConstrainedInt', type.__new__(cls, name, bases, dct))
        if new_cls.gt is not None and new_cls.ge is not None:
            raise errors.ConfigError('bounds gt and ge cannot be specified at the same time')
        if new_cls.lt is not None and new_cls.le is not None:
            raise errors.ConfigError('bounds lt and le cannot be specified at the same time')
        return _registered(new_cls)