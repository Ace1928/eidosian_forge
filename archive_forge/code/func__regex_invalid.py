from __future__ import annotations
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping as _Mapping
from functools import lru_cache
from typing import Union, Mapping, overload
from numbers import Complex
import numpy as np
from numpy.typing import ArrayLike
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from .object_array import object_array
from .shape import ShapedMixin, shape_tuple
@lru_cache(1)
def _regex_invalid(allowed_chars: str) -> re.Pattern:
    """Return pattern for selecting invalid strings"""
    return re.compile(f'[^{re.escape(allowed_chars)}]')