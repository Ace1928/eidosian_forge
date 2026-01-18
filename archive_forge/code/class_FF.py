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
class FF(SparsePrecon):
    """Creates matrix using morse/bond/angle/dihedral force field parameters.
    """

    def __init__(self, dim=3, c_stab=0.1, force_stab=False, array_convention='C', solver='auto', solve_tol=1e-09, apply_positions=True, apply_cell=True, hessian='spectral', morses=None, bonds=None, angles=None, dihedrals=None, logfile=None):
        """Initialise an FF preconditioner with given parameters.

        Args:
             dim, c_stab, force_stab, array_convention, use_pyamg, solve_tol:
                see SparsePrecon.__init__()
             morses: Morse instance
             bonds: Bond instance
             angles: Angle instance
             dihedrals: Dihedral instance
        """
        if morses is None and bonds is None and (angles is None) and (dihedrals is None):
            raise ImportError('At least one of morses, bonds, angles or dihedrals must be defined!')
        super().__init__(dim=dim, c_stab=c_stab, force_stab=force_stab, array_convention=array_convention, solver=solver, solve_tol=solve_tol, apply_positions=apply_positions, apply_cell=apply_cell, logfile=logfile)
        self.hessian = hessian
        self.morses = morses
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals

    def make_precon(self, atoms, reinitialize=None):
        start_time = time.time()
        self._make_sparse_precon(atoms, force_stab=self.force_stab)
        self.logfile.write('--- Precon created in %s seconds ---\n' % (time.time() - start_time))

    def add_morse(self, morse, atoms, row, col, data, conn=None):
        if self.hessian == 'reduced':
            i, j, Hx = ff.get_morse_potential_reduced_hessian(atoms, morse)
        elif self.hessian == 'spectral':
            i, j, Hx = ff.get_morse_potential_hessian(atoms, morse, spectral=True)
        else:
            raise NotImplementedError('Not implemented hessian')
        x = ij_to_x(i, j)
        row.extend(np.repeat(x, 6))
        col.extend(np.tile(x, 6))
        data.extend(Hx.flatten())
        if conn is not None:
            conn[i, j] = True
            conn[j, i] = True

    def add_bond(self, bond, atoms, row, col, data, conn=None):
        if self.hessian == 'reduced':
            i, j, Hx = ff.get_bond_potential_reduced_hessian(atoms, bond, self.morses)
        elif self.hessian == 'spectral':
            i, j, Hx = ff.get_bond_potential_hessian(atoms, bond, self.morses, spectral=True)
        else:
            raise NotImplementedError('Not implemented hessian')
        x = ij_to_x(i, j)
        row.extend(np.repeat(x, 6))
        col.extend(np.tile(x, 6))
        data.extend(Hx.flatten())
        if conn is not None:
            conn[i, j] = True
            conn[j, i] = True

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

    def _make_sparse_precon(self, atoms, initial_assembly=False, force_stab=False):
        N = len(atoms)
        row = []
        col = []
        data = []
        if self.morses is not None:
            for morse in self.morses:
                self.add_morse(morse, atoms, row, col, data)
        if self.bonds is not None:
            for bond in self.bonds:
                self.add_bond(bond, atoms, row, col, data)
        if self.angles is not None:
            for angle in self.angles:
                self.add_angle(angle, atoms, row, col, data)
        if self.dihedrals is not None:
            for dihedral in self.dihedrals:
                self.add_dihedral(dihedral, atoms, row, col, data)
        row.extend(range(self.dim * N))
        col.extend(range(self.dim * N))
        data.extend([self.c_stab] * self.dim * N)
        start_time = time.time()
        self.P = sparse.csc_matrix((data, (row, col)), shape=(self.dim * N, self.dim * N))
        self.logfile.write('--- created CSC matrix in %s s ---\n' % (time.time() - start_time))
        self.P = apply_fixed(atoms, self.P)
        self.P = self.P.tocsr()
        self.logfile.write('--- N-dim precon created in %s s ---\n' % (time.time() - start_time))
        self.create_solver()