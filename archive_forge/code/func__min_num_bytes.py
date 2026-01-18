from __future__ import annotations
from collections import defaultdict
from functools import partial
from itertools import chain, repeat
from typing import Callable, Iterable, Literal, Mapping
import numpy as np
from numpy.typing import NDArray
from qiskit.result import Counts
from .shape import ShapedMixin, ShapeInput, shape_tuple
def _min_num_bytes(num_bits: int) -> int:
    """Return the minimum number of bytes needed to store ``num_bits``."""
    return num_bits // 8 + (num_bits % 8 > 0)