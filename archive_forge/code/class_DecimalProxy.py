import re
import math
from decimal import Decimal
from typing import Any, Union, SupportsFloat
from ..helpers import BOOLEAN_VALUES, collapse_white_spaces, get_double
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
from .numeric import Float10, Integer
from .datetime import AbstractDateTime, Duration
class DecimalProxy(AnyAtomicType):
    name = 'decimal'
    pattern = re.compile('^[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)$')

    def __new__(cls, value: Any) -> Decimal:
        if isinstance(value, (str, UntypedAtomic)):
            value = collapse_white_spaces(str(value)).replace(' ', '')
            if cls.pattern.match(value) is None:
                raise cls.invalid_value(value)
        elif isinstance(value, (float, Float10, Decimal)):
            if math.isinf(value) or math.isnan(value):
                raise cls.invalid_value(value)
        try:
            return Decimal(value)
        except (ValueError, ArithmeticError):
            msg = 'invalid value {!r} for xs:{}'
            raise ArithmeticError(msg.format(value, cls.name)) from None

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        return issubclass(subclass, (int, Decimal, Integer)) and (not issubclass(subclass, bool))

    @classmethod
    def validate(cls, value: object) -> None:
        if isinstance(value, Decimal):
            if math.isnan(value) or math.isinf(value):
                raise cls.invalid_value(value)
        elif isinstance(value, (int, Integer)) and (not isinstance(value, bool)):
            return
        elif isinstance(value, str):
            if cls.pattern.match(value) is None:
                raise cls.invalid_value(value)
        else:
            raise cls.invalid_type(value)