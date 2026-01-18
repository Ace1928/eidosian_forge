import posixpath as pp
import sys
import numpy
from .. import h5, h5s, h5t, h5r, h5d, h5p, h5fd, h5ds, _selector
from .base import (
from . import filters
from . import selections as sel
from . import selections2 as sel2
from .datatype import Datatype
from .compat import filename_decode
from .vds import VDSmap, vds_support
class ChunkIterator:
    """
    Class to iterate through list of chunks of a given dataset
    """

    def __init__(self, dset, source_sel=None):
        self._shape = dset.shape
        rank = len(dset.shape)
        if not dset.chunks:
            raise TypeError('Chunked dataset required')
        self._layout = dset.chunks
        if source_sel is None:
            slices = []
            for dim in range(rank):
                slices.append(slice(0, self._shape[dim]))
            self._sel = tuple(slices)
        elif isinstance(source_sel, slice):
            self._sel = (source_sel,)
        else:
            self._sel = source_sel
        if len(self._sel) != rank:
            raise ValueError('Invalid selection - selection region must have same rank as dataset')
        self._chunk_index = []
        for dim in range(rank):
            s = self._sel[dim]
            if s.start < 0 or s.stop > self._shape[dim] or s.stop <= s.start:
                raise ValueError('Invalid selection - selection region must be within dataset space')
            index = s.start // self._layout[dim]
            self._chunk_index.append(index)

    def __iter__(self):
        return self

    def __next__(self):
        rank = len(self._shape)
        slices = []
        if rank == 0 or self._chunk_index[0] * self._layout[0] >= self._sel[0].stop:
            raise StopIteration()
        for dim in range(rank):
            s = self._sel[dim]
            start = self._chunk_index[dim] * self._layout[dim]
            stop = (self._chunk_index[dim] + 1) * self._layout[dim]
            if start < s.start:
                start = s.start
            if stop > s.stop:
                stop = s.stop
            s = slice(start, stop, 1)
            slices.append(s)
        dim = rank - 1
        while dim >= 0:
            s = self._sel[dim]
            self._chunk_index[dim] += 1
            chunk_end = self._chunk_index[dim] * self._layout[dim]
            if chunk_end < s.stop:
                return tuple(slices)
            if dim > 0:
                self._chunk_index[dim] = 0
            dim -= 1
        return tuple(slices)