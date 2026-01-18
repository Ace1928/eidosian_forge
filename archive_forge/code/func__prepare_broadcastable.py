from __future__ import annotations
from collections import defaultdict
from functools import partial
from itertools import chain, repeat
from typing import Callable, Iterable, Literal, Mapping
import numpy as np
from numpy.typing import NDArray
from qiskit.result import Counts
from .shape import ShapedMixin, ShapeInput, shape_tuple
def _prepare_broadcastable(self, other: 'BitArray') -> tuple[NDArray[np.uint8], ...]:
    """Validation and broadcasting of two bit arrays before element-wise binary operation."""
    if self.num_bits != other.num_bits:
        raise ValueError(f"'num_bits' must match in {self} and {other}.")
    self_shape = self.shape + (self.num_shots,)
    other_shape = other.shape + (other.num_shots,)
    try:
        shape = np.broadcast_shapes(self_shape, other_shape) + (self._array.shape[-1],)
    except ValueError as ex:
        raise ValueError(f'{self} and {other} are not compatible for this operation.') from ex
    return (np.broadcast_to(self.array, shape), np.broadcast_to(other.array, shape))