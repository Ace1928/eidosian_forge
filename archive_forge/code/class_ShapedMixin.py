from __future__ import annotations
from collections.abc import Iterable
from typing import Protocol, Union, runtime_checkable
import numpy as np
from numpy.typing import ArrayLike, NDArray
class ShapedMixin(Shaped):
    """Mixin class to create :class:`~Shaped` types by only providing :attr:`_shape` attribute."""
    _shape: tuple[int, ...]

    def __repr__(self):
        return f'{type(self).__name__}(<{self.shape}>)'

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return int(np.prod(self._shape, dtype=int))