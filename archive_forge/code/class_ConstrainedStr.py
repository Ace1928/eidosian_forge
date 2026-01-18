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
class ConstrainedStr(str):
    strip_whitespace = False
    to_upper = False
    to_lower = False
    min_length: OptionalInt = None
    max_length: OptionalInt = None
    curtail_length: OptionalInt = None
    regex: Optional[Union[str, Pattern[str]]] = None
    strict = False

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minLength=cls.min_length, maxLength=cls.max_length, pattern=cls.regex and cls._get_pattern(cls.regex))

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield (strict_str_validator if cls.strict else str_validator)
        yield constr_strip_whitespace
        yield constr_upper
        yield constr_lower
        yield constr_length_validator
        yield cls.validate

    @classmethod
    def validate(cls, value: Union[str]) -> Union[str]:
        if cls.curtail_length and len(value) > cls.curtail_length:
            value = value[:cls.curtail_length]
        if cls.regex:
            if not re.match(cls.regex, value):
                raise errors.StrRegexError(pattern=cls._get_pattern(cls.regex))
        return value

    @staticmethod
    def _get_pattern(regex: Union[str, Pattern[str]]) -> str:
        return regex if isinstance(regex, str) else regex.pattern