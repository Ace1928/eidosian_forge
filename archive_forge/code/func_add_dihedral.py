import sys
import time
import copy
import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from ase.constraints import Filter, FixAtoms
from ase.utils import longsum
from ase.geometry import find_mic
import ase.utils.ff as ff
import ase.units as units
from ase.optimize.precon.neighbors import (get_neighbours,
from ase.neighborlist import neighbor_list
def add_dihedral(self, dihedral, atoms, row, col, data, conn=None):
    if self.hessian == 'reduced':
        i, j, k, l, Hx = ff.get_dihedral_potential_reduced_hessian(atoms, dihedral, self.morses)
    elif self.hessian == 'spectral':
        i, j, k, l, Hx = ff.get_dihedral_potential_hessian(atoms, dihedral, self.morses, spectral=True)
    else:
        raise NotImplementedError('Not implemented hessian')
    x = ijkl_to_x(i, j, k, l)
    row.extend(np.repeat(x, 12))
    col.extend(np.tile(x, 12))
    data.extend(Hx.flatten())
    if conn is not None:
        conn[i, j] = conn[i, k] = conn[i, l] = conn[j, k] = conn[j, l] = conn[k, l] = True
        conn[j, i] = conn[k, i] = conn[l, i] = conn[k, j] = conn[l, j] = conn[l, k] = True