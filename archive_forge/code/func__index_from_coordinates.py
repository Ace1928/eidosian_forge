import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def _index_from_coordinates(s: Sequence[int], volume: Sequence[int]) -> int:
    return np.dot(s, volume)