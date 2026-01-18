import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
def _full_join_string_lists(list1: Optional[Sequence[str]], list2: Optional[Sequence[str]]) -> Optional[Sequence[str]]:
    if list1 is None and list2 is None:
        return None
    if list1 is None:
        return list2
    if list2 is None:
        return list1
    return [f'{first}{REPETITION_ID_SEPARATOR}{second}' for first in list1 for second in list2]