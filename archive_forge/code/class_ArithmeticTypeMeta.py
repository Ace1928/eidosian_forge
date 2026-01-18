import re
import math
from decimal import Decimal
from typing import Any, Union, SupportsFloat
from ..helpers import BOOLEAN_VALUES, collapse_white_spaces, get_double
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
from .numeric import Float10, Integer
from .datetime import AbstractDateTime, Duration
class ArithmeticTypeMeta(type):
    """Metaclass for checking numeric, datetime and duration classes/instances."""

    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, (int, float, Decimal, AbstractDateTime, Duration, UntypedAtomic)) and (not isinstance(instance, bool))

    def __subclasscheck__(cls, subclass: type) -> bool:
        if issubclass(subclass, bool):
            return False
        return issubclass(subclass, int) or issubclass(subclass, float) or issubclass(subclass, Decimal) or issubclass(subclass, Duration) or issubclass(subclass, AbstractDateTime) or issubclass(subclass, UntypedAtomic)