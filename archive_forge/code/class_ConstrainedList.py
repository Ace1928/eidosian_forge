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
class ConstrainedList(list):
    __origin__ = list
    __args__: Tuple[Type[T], ...]
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    unique_items: Optional[bool] = None
    item_type: Type[T]

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.list_length_validator
        if cls.unique_items:
            yield cls.unique_items_validator

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minItems=cls.min_items, maxItems=cls.max_items, uniqueItems=cls.unique_items)

    @classmethod
    def list_length_validator(cls, v: 'Optional[List[T]]') -> 'Optional[List[T]]':
        if v is None:
            return None
        v = list_validator(v)
        v_len = len(v)
        if cls.min_items is not None and v_len < cls.min_items:
            raise errors.ListMinLengthError(limit_value=cls.min_items)
        if cls.max_items is not None and v_len > cls.max_items:
            raise errors.ListMaxLengthError(limit_value=cls.max_items)
        return v

    @classmethod
    def unique_items_validator(cls, v: 'Optional[List[T]]') -> 'Optional[List[T]]':
        if v is None:
            return None
        for i, value in enumerate(v, start=1):
            if value in v[i:]:
                raise errors.ListUniqueItemsError()
        return v