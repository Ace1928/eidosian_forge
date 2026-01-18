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
class ConstrainedBytes(bytes):
    strip_whitespace = False
    to_upper = False
    to_lower = False
    min_length: OptionalInt = None
    max_length: OptionalInt = None
    strict: bool = False

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minLength=cls.min_length, maxLength=cls.max_length)

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield (strict_bytes_validator if cls.strict else bytes_validator)
        yield constr_strip_whitespace
        yield constr_upper
        yield constr_lower
        yield constr_length_validator