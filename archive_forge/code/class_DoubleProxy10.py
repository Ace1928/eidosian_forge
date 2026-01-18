import re
import math
from decimal import Decimal
from typing import Any, Union, SupportsFloat
from ..helpers import BOOLEAN_VALUES, collapse_white_spaces, get_double
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
from .numeric import Float10, Integer
from .datetime import AbstractDateTime, Duration
class DoubleProxy10(AnyAtomicType):
    name = 'double'
    xsd_version = '1.0'
    pattern = re.compile('^(?:[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[Ee][+-]?[0-9]+)?|[+-]?INF|NaN)$')

    def __new__(cls, value: Union[SupportsFloat, str]) -> float:
        return get_double(value, cls.xsd_version)

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        return issubclass(subclass, float) and (not issubclass(subclass, Float10))

    @classmethod
    def validate(cls, value: object) -> None:
        if isinstance(value, float) and (not isinstance(value, Float10)):
            return
        elif isinstance(value, str):
            if cls.pattern.match(value) is None:
                raise cls.invalid_value(value)
        else:
            raise cls.invalid_type(value)