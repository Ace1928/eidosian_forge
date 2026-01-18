import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
import numpy
from .compat import cupy, has_cupy
@dataclass
class Ragged:
    """A batch of concatenated sequences, that vary in the size of their
    first dimension. Ragged allows variable-length sequence data to be contiguous
    in memory, without padding.

    Indexing into Ragged is just like indexing into the *lengths* array, except
    it returns a Ragged object with the accompanying sequence data. For instance,
    you can write ragged[1:4] to get a Ragged object with sequences 1, 2 and 3.
    """
    data: Array2d
    lengths: Ints1d
    data_shape: Tuple[int, ...]
    starts_ends: Optional[Ints1d] = None

    def __init__(self, data: _Array, lengths: Ints1d):
        self.lengths = lengths
        if data.size:
            self.data = cast(Array2d, data.reshape((data.shape[0], -1)))
        else:
            self.data = cast(Array2d, data.reshape((0, 0)))
        self.data_shape = (-1,) + data.shape[1:]

    @property
    def dataXd(self) -> ArrayXd:
        if self.data.size:
            reshaped = self.data.reshape(self.data_shape)
        else:
            reshaped = self.data.reshape((self.data.shape[0],) + self.data_shape[1:])
        return cast(ArrayXd, reshaped)

    def __len__(self) -> int:
        return self.lengths.shape[0]

    def __getitem__(self, index: Union[int, slice, Array1d]) -> 'Ragged':
        if isinstance(index, tuple):
            raise IndexError('Ragged arrays do not support 2d indexing.')
        starts = self._get_starts()
        ends = self._get_ends()
        if isinstance(index, int):
            s = starts[index]
            e = ends[index]
            return Ragged(self.data[s:e], self.lengths[index:index + 1])
        elif isinstance(index, slice):
            lengths = self.lengths[index]
            if len(lengths) == 0:
                return Ragged(self.data[0:0].reshape(self.data_shape), lengths)
            start = starts[index][0] if index.start >= 1 else 0
            end = ends[index][-1]
            return Ragged(self.data[start:end].reshape(self.data_shape), lengths)
        else:
            xp = get_array_module(self.data)
            data = xp.vstack([self[int(i)].data for i in index])
            return Ragged(data.reshape(self.data_shape), self.lengths[index])

    def _get_starts_ends(self) -> Ints1d:
        if self.starts_ends is None:
            xp = get_array_module(self.lengths)
            self.starts_ends = xp.empty(self.lengths.size + 1, dtype='i')
            self.starts_ends[0] = 0
            self.lengths.cumsum(out=self.starts_ends[1:])
        return self.starts_ends

    def _get_starts(self) -> Ints1d:
        return self._get_starts_ends()[:-1]

    def _get_ends(self) -> Ints1d:
        return self._get_starts_ends()[1:]