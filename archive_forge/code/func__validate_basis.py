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
@classmethod
def _validate_basis(cls, basis: str) -> None:
    """Validate a basis string.

        Args:
            basis: a basis string to validate.

        Raises:
            ValueError: If basis string contains invalid characters
        """
    allowed_pattern = _regex_match(cls.ALLOWED_BASIS)
    if not allowed_pattern.match(basis):
        invalid_pattern = _regex_invalid(cls.ALLOWED_BASIS)
        invalid_chars = list(set(invalid_pattern.findall(basis)))
        raise ValueError(f"Observable basis string '{basis}' contains invalid characters {invalid_chars}, allowed characters are {list(cls.ALLOWED_BASIS)}.")