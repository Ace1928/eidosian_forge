from typing import AbstractSet, Any, Dict, Optional, Tuple, TYPE_CHECKING, Union, List
import datetime
import sympy
import numpy as np
from cirq import protocols
from cirq._compat import proper_repr, cached_method
from cirq._doc import document
def _attempt_duration_like_to_duration(value: Any) -> Optional[Duration]:
    if isinstance(value, Duration):
        return value
    if isinstance(value, datetime.timedelta):
        return Duration(value)
    if isinstance(value, (int, float)) and value == 0:
        return Duration()
    return None