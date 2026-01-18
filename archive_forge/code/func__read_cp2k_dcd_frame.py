import numpy as np
from itertools import islice
import os
from ase.atoms import Atoms, Atom
from ase.cell import Cell
from ase.io.formats import index2range
from ase.data import atomic_numbers
def _read_cp2k_dcd_frame(fileobj, dtype, natoms, symbols, aligned=False):
    arr = np.fromfile(fileobj, dtype, 1)
    cryst_const = np.empty(6, dtype=np.float64)
    cryst_const[0] = arr['x1'][0, 0]
    cryst_const[1] = arr['x1'][0, 2]
    cryst_const[2] = arr['x1'][0, 5]
    cryst_const[3] = arr['x1'][0, 4]
    cryst_const[4] = arr['x1'][0, 3]
    cryst_const[5] = arr['x1'][0, 1]
    coords = np.empty((natoms, 3), dtype=np.float32)
    coords[..., 0] = arr['x3'][0, ...]
    coords[..., 1] = arr['x5'][0, ...]
    coords[..., 2] = arr['x7'][0, ...]
    assert len(coords) == len(symbols)
    if aligned:
        atoms = Atoms(symbols, coords, cell=cryst_const, pbc=True)
    else:
        atoms = Atoms(symbols, coords)
    return atoms