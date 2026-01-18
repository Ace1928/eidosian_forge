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
class ConstrainedDecimal(Decimal, metaclass=ConstrainedNumberMeta):
    gt: OptionalIntFloatDecimal = None
    ge: OptionalIntFloatDecimal = None
    lt: OptionalIntFloatDecimal = None
    le: OptionalIntFloatDecimal = None
    max_digits: OptionalInt = None
    decimal_places: OptionalInt = None
    multiple_of: OptionalIntFloatDecimal = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, exclusiveMinimum=cls.gt, exclusiveMaximum=cls.lt, minimum=cls.ge, maximum=cls.le, multipleOf=cls.multiple_of)

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield decimal_validator
        yield number_size_validator
        yield number_multiple_validator
        yield cls.validate

    @classmethod
    def validate(cls, value: Decimal) -> Decimal:
        try:
            normalized_value = value.normalize()
        except InvalidOperation:
            normalized_value = value
        digit_tuple, exponent = normalized_value.as_tuple()[1:]
        if exponent in {'F', 'n', 'N'}:
            raise errors.DecimalIsNotFiniteError()
        if exponent >= 0:
            digits = len(digit_tuple) + exponent
            decimals = 0
        elif abs(exponent) > len(digit_tuple):
            digits = decimals = abs(exponent)
        else:
            digits = len(digit_tuple)
            decimals = abs(exponent)
        whole_digits = digits - decimals
        if cls.max_digits is not None and digits > cls.max_digits:
            raise errors.DecimalMaxDigitsError(max_digits=cls.max_digits)
        if cls.decimal_places is not None and decimals > cls.decimal_places:
            raise errors.DecimalMaxPlacesError(decimal_places=cls.decimal_places)
        if cls.max_digits is not None and cls.decimal_places is not None:
            expected = cls.max_digits - cls.decimal_places
            if whole_digits > expected:
                raise errors.DecimalWholeDigitsError(whole_digits=expected)
        return value