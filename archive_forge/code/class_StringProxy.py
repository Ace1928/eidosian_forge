import re
import math
from decimal import Decimal
from typing import Any, Union, SupportsFloat
from ..helpers import BOOLEAN_VALUES, collapse_white_spaces, get_double
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
from .numeric import Float10, Integer
from .datetime import AbstractDateTime, Duration
class StringProxy(AnyAtomicType):
    name = 'string'

    def __new__(cls, *args: object, **kwargs: object) -> str:
        return str(*args, **kwargs)

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        return issubclass(subclass, str)

    @classmethod
    def validate(cls, value: object) -> None:
        if not isinstance(value, str):
            raise cls.invalid_type(value)