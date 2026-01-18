import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def _coordinates_from_index(idx: int, volume: Sequence[int]) -> Sequence[int]:
    ret = []
    for v in volume:
        ret.append(idx // v)
        idx %= v
    return tuple(ret)