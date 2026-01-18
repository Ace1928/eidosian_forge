import numpy as np
from math import sqrt
from itertools import islice
from ase.io.formats import string2index
from ase.utils import rotate
from ase.data import covalent_radii, atomic_numbers
from ase.data.colors import jmol_colors
class ImageIterator:
    """Iterate over chunks, to return the corresponding Atoms objects.
    Will only build the atoms objects which corresponds to the requested
    indices when called.
    Assumes ``ichunks`` is in iterator, which returns ``ImageChunk``
    type objects. See extxyz.py:iread_xyz as an example.
    """

    def __init__(self, ichunks):
        self.ichunks = ichunks

    def __call__(self, fd, index=None, **kwargs):
        if isinstance(index, str):
            index = string2index(index)
        if index is None or index == ':':
            index = slice(None, None, None)
        if not isinstance(index, (slice, str)):
            index = slice(index, index + 1 or None)
        for chunk in self._getslice(fd, index):
            yield chunk.build(**kwargs)

    def _getslice(self, fd, indices):
        try:
            iterator = islice(self.ichunks(fd), indices.start, indices.stop, indices.step)
        except ValueError:
            if not hasattr(fd, 'seekable') or not fd.seekable():
                raise ValueError('Negative indices only supported for seekable streams')
            startpos = fd.tell()
            nchunks = 0
            for chunk in self.ichunks(fd):
                nchunks += 1
            fd.seek(startpos)
            indices_tuple = indices.indices(nchunks)
            iterator = islice(self.ichunks(fd), *indices_tuple)
        return iterator