import re
import math
from decimal import Decimal
from typing import Any, Union, SupportsFloat
from ..helpers import BOOLEAN_VALUES, collapse_white_spaces, get_double
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
from .numeric import Float10, Integer
from .datetime import AbstractDateTime, Duration
class BooleanProxy(AnyAtomicType):
    name = 'boolean'
    pattern = re.compile('^(?:true|false|1|0)$')

    def __new__(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        elif isinstance(value, (int, float, Decimal)):
            if math.isnan(value):
                return False
            return bool(value)
        elif isinstance(value, UntypedAtomic):
            value = value.value
        elif not isinstance(value, str):
            raise TypeError('invalid type {!r} for xs:{}'.format(type(value), cls.name))
        if value.strip() not in BOOLEAN_VALUES:
            raise ValueError('invalid value {!r} for xs:{}'.format(value, cls.name))
        return 't' in value or '1' in value

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        return issubclass(subclass, bool)

    @classmethod
    def validate(cls, value: object) -> None:
        if isinstance(value, bool):
            return
        elif isinstance(value, str):
            if cls.pattern.match(value) is None:
                raise cls.invalid_value(value)
        else:
            raise cls.invalid_type(value)