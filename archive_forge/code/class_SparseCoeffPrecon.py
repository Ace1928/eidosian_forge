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
class SparseCoeffPrecon(SparsePrecon):

    def _make_sparse_precon(self, atoms, initial_assembly=False, force_stab=False):
        """Create a sparse preconditioner matrix based on the passed atoms.

        Creates a general-purpose preconditioner for use with optimization
        algorithms, based on examining distances between pairs of atoms in the
        lattice. The matrix will be stored in the attribute self.P and
        returned. Note that this function will use self.mu, whatever it is.

        Args:
            atoms: the Atoms object used to create the preconditioner.

        Returns:
            A scipy.sparse.csr_matrix object, representing a d*N by d*N matrix
            (where N is the number of atoms, and d is the value of self.dim).
            BE AWARE that using numpy.dot() with this object will result in
            errors/incorrect results - use the .dot method directly on the
            sparse matrix instead.

        """
        logfile = self.logfile
        logfile.write('creating sparse precon: initial_assembly=%r, force_stab=%r, apply_positions=%r, apply_cell=%r\n' % (initial_assembly, force_stab, self.apply_positions, self.apply_cell))
        N = len(atoms)
        diag_i = np.arange(N, dtype=int)
        start_time = time.time()
        if self.apply_positions:
            i, j, rij, fixed_atoms = get_neighbours(atoms, self.r_cut, neighbor_list=self.neighbor_list)
            logfile.write('--- neighbour list created in %s s --- \n' % (time.time() - start_time))
            start_time = time.time()
            coeff = self.get_coeff(rij)
            diag_coeff = np.bincount(i, -coeff, minlength=N).astype(np.float64)
            if force_stab or len(fixed_atoms) == 0:
                logfile.write('adding stabilisation to precon')
                diag_coeff += self.mu * self.c_stab
        else:
            diag_coeff = np.ones(N)
        if isinstance(atoms, Filter):
            if self.apply_cell:
                diag_coeff[-3:] = self.mu_c
            else:
                diag_coeff[-3:] = 1.0
        logfile.write('--- computed triplet format in %s s ---\n' % (time.time() - start_time))
        if self.apply_positions and (not initial_assembly):
            start_time = time.time()
            mask = np.ones(N)
            mask[fixed_atoms] = 0.0
            coeff *= mask[i] * mask[j]
            diag_coeff[fixed_atoms] = 1.0
            logfile.write('--- applied fixed_atoms in %s s ---\n' % (time.time() - start_time))
        if self.apply_positions:
            start_time = time.time()
            inz = np.nonzero(coeff)
            i = np.hstack((i[inz], diag_i))
            j = np.hstack((j[inz], diag_i))
            coeff = np.hstack((coeff[inz], diag_coeff))
            logfile.write('--- remove zeros in %s s ---\n' % (time.time() - start_time))
        else:
            i = diag_i
            j = diag_i
            coeff = diag_coeff
        start_time = time.time()
        csc_P = sparse.csc_matrix((coeff, (i, j)), shape=(N, N))
        logfile.write('--- created CSC matrix in %s s ---\n' % (time.time() - start_time))
        self.P = self.one_dim_to_ndim(csc_P, N)
        self.create_solver()

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
        self.logfile.write('--- Precon created in %s seconds --- \n' % (time.time() - start_time))

    @abstractmethod
    def get_coeff(self, r):
        ...