import numpy as np
from itertools import islice
import os
from ase.atoms import Atoms, Atom
from ase.cell import Cell
from ase.io.formats import index2range
from ase.data import atomic_numbers
def _getslice(self, fd, indices):
    try:
        iterator = islice(self.ichunks(fd, self.ref_atoms, self.aligned), indices.start, indices.stop, indices.step)
    except ValueError:
        dtype, natoms, nsteps, header_end = _read_metainfo(fd)
        indices_tuple = indices.indices(nsteps + 1)
        iterator = islice(self.ichunks(fd, self.ref_atoms, self.aligned), *indices_tuple)
    return iterator