from typing import Sequence, Union, List, Iterator, TYPE_CHECKING, Iterable, Optional
import numpy as np
from cirq import circuits, devices, linalg, ops, protocols
def _sticky_0_to_1(v: float, *, atol: float) -> Optional[float]:
    if 0 <= v <= 1:
        return v
    if 1 < v <= 1 + atol:
        return 1
    if 0 > v >= -atol:
        return 0
    return None