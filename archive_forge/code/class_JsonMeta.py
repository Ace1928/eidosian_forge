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
class JsonMeta(type):

    def __getitem__(self, t: Type[Any]) -> Type[JsonWrapper]:
        if t is Any:
            return Json
        return _registered(type('JsonWrapperValue', (JsonWrapper,), {'inner_type': t}))