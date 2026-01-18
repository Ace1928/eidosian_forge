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
def add_angle(self, angle, atoms, row, col, data, conn=None):
    if self.hessian == 'reduced':
        i, j, k, Hx = ff.get_angle_potential_reduced_hessian(atoms, angle, self.morses)
    elif self.hessian == 'spectral':
        i, j, k, Hx = ff.get_angle_potential_hessian(atoms, angle, self.morses, spectral=True)
    else:
        raise NotImplementedError('Not implemented hessian')
    x = ijk_to_x(i, j, k)
    row.extend(np.repeat(x, 9))
    col.extend(np.tile(x, 9))
    data.extend(Hx.flatten())
    if conn is not None:
        conn[i, j] = conn[i, k] = conn[j, k] = True
        conn[j, i] = conn[k, i] = conn[k, j] = True