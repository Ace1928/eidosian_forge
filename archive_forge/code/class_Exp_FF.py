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
class Exp_FF(Exp, FF):
    """Creates matrix with values decreasing exponentially with distance.
    """

    def __init__(self, A=3.0, r_cut=None, r_NN=None, mu=None, mu_c=None, dim=3, c_stab=0.1, force_stab=False, reinitialize=False, array_convention='C', solver='auto', solve_tol=1e-09, apply_positions=True, apply_cell=True, estimate_mu_eigmode=False, hessian='spectral', morses=None, bonds=None, angles=None, dihedrals=None, logfile=None):
        """Initialise an Exp+FF preconditioner with given parameters.

        Args:
            r_cut, mu, c_stab, dim, reinitialize, array_convention: see
                precon.__init__()
            A: coefficient in exp(-A*r/r_NN). Default is A=3.0.
        """
        if morses is None and bonds is None and (angles is None) and (dihedrals is None):
            raise ImportError('At least one of morses, bonds, angles or dihedrals must be defined!')
        SparsePrecon.__init__(self, r_cut=r_cut, r_NN=r_NN, mu=mu, mu_c=mu_c, dim=dim, c_stab=c_stab, force_stab=force_stab, reinitialize=reinitialize, array_convention=array_convention, solver=solver, solve_tol=solve_tol, apply_positions=apply_positions, apply_cell=apply_cell, estimate_mu_eigmode=estimate_mu_eigmode, logfile=logfile)
        self.A = A
        self.hessian = hessian
        self.morses = morses
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals

    def make_precon(self, atoms, reinitialize=None):
        if self.r_NN is None:
            self.r_NN = estimate_nearest_neighbour_distance(atoms, self.neighbor_list)
        if self.r_cut is None:
            self.r_cut = 2.0 * self.r_NN
        elif self.r_cut < self.r_NN:
            warning = 'WARNING: r_cut (%.2f) < r_NN (%.2f), increasing to 1.1*r_NN = %.2f' % (self.r_cut, self.r_NN, 1.1 * self.r_NN)
            warnings.warn(warning)
            self.r_cut = 1.1 * self.r_NN
        if reinitialize is None:
            reinitialize = self.reinitialize
        if self.mu is None:
            reinitialize = True
        if reinitialize:
            self.estimate_mu(atoms)
        if self.P is not None:
            real_atoms = atoms
            if isinstance(atoms, Filter):
                real_atoms = atoms.atoms
            if self.old_positions is None:
                self.old_positions = real_atoms.positions
            displacement, _ = find_mic(real_atoms.positions - self.old_positions, real_atoms.cell, real_atoms.pbc)
            self.old_positions = real_atoms.get_positions()
            max_abs_displacement = abs(displacement).max()
            self.logfile.write('max(abs(displacements)) = %.2f A (%.2f r_NN)' % (max_abs_displacement, max_abs_displacement / self.r_NN))
            if max_abs_displacement < 0.5 * self.r_NN:
                return
        start_time = time.time()
        self._make_sparse_precon(atoms, force_stab=self.force_stab)
        self.logfile.write('--- Precon created in %s seconds ---\n' % (time.time() - start_time))

    def _make_sparse_precon(self, atoms, initial_assembly=False, force_stab=False):
        """Create a sparse preconditioner matrix based on the passed atoms.

        Args:
            atoms: the Atoms object used to create the preconditioner.

        Returns:
            A scipy.sparse.csr_matrix object, representing a d*N by d*N matrix
            (where N is the number of atoms, and d is the value of self.dim).
            BE AWARE that using numpy.dot() with this object will result in
            errors/incorrect results - use the .dot method directly on the
            sparse matrix instead.

        """
        self.logfile.write('creating sparse precon: initial_assembly=%r, force_stab=%r, apply_positions=%r, apply_cell=%r\n' % (initial_assembly, force_stab, self.apply_positions, self.apply_cell))
        N = len(atoms)
        start_time = time.time()
        if self.apply_positions:
            i_list, j_list, rij_list, fixed_atoms = get_neighbours(atoms, self.r_cut, self.neighbor_list)
            self.logfile.write('--- neighbour list created in %s s ---\n' % (time.time() - start_time))
        row = []
        col = []
        data = []
        start_time = time.time()
        if isinstance(atoms, Filter):
            i = N - 3
            j = N - 2
            k = N - 1
            x = ijk_to_x(i, j, k)
            row.extend(x)
            col.extend(x)
            if self.apply_cell:
                data.extend(np.repeat(self.mu_c, 9))
            else:
                data.extend(np.repeat(self.mu_c, 9))
        self.logfile.write('--- computed triplet format in %s s ---\n' % (time.time() - start_time))
        conn = sparse.lil_matrix((N, N), dtype=bool)
        if self.apply_positions and (not initial_assembly):
            if self.morses is not None:
                for morse in self.morses:
                    self.add_morse(morse, atoms, row, col, data, conn)
            if self.bonds is not None:
                for bond in self.bonds:
                    self.add_bond(bond, atoms, row, col, data, conn)
            if self.angles is not None:
                for angle in self.angles:
                    self.add_angle(angle, atoms, row, col, data, conn)
            if self.dihedrals is not None:
                for dihedral in self.dihedrals:
                    self.add_dihedral(dihedral, atoms, row, col, data, conn)
        if self.apply_positions:
            for i, j, rij in zip(i_list, j_list, rij_list):
                if not conn[i, j]:
                    coeff = self.get_coeff(rij)
                    x = ij_to_x(i, j)
                    row.extend(x)
                    col.extend(x)
                    data.extend(3 * [-coeff] + 3 * [coeff])
        row.extend(range(self.dim * N))
        col.extend(range(self.dim * N))
        if initial_assembly:
            data.extend([self.mu * self.c_stab] * self.dim * N)
        else:
            data.extend([self.c_stab] * self.dim * N)
        start_time = time.time()
        self.P = sparse.csc_matrix((data, (row, col)), shape=(self.dim * N, self.dim * N))
        self.logfile.write('--- created CSC matrix in %s s ---\n' % (time.time() - start_time))
        if not initial_assembly:
            self.P = apply_fixed(atoms, self.P)
        self.P = self.P.tocsr()
        self.create_solver()