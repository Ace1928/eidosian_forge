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
class ConstrainedInt(int, metaclass=ConstrainedNumberMeta):
    strict: bool = False
    gt: OptionalInt = None
    ge: OptionalInt = None
    lt: OptionalInt = None
    le: OptionalInt = None
    multiple_of: OptionalInt = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, exclusiveMinimum=cls.gt, exclusiveMaximum=cls.lt, minimum=cls.ge, maximum=cls.le, multipleOf=cls.multiple_of)

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield (strict_int_validator if cls.strict else int_validator)
        yield number_size_validator
        yield number_multiple_validator