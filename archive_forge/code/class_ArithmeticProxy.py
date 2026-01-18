import re
import math
from decimal import Decimal
from typing import Any, Union, SupportsFloat
from ..helpers import BOOLEAN_VALUES, collapse_white_spaces, get_double
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
from .numeric import Float10, Integer
from .datetime import AbstractDateTime, Duration
class ArithmeticProxy(metaclass=ArithmeticTypeMeta):
    """Proxy for arithmetic related types. Builds xs:float instances."""

    def __new__(cls, *args: FloatArgType, **kwargs: FloatArgType) -> float:
        return float(*args, **kwargs)