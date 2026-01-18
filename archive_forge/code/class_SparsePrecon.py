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
class SparsePrecon(Precon):

    def __init__(self, r_cut=None, r_NN=None, mu=None, mu_c=None, dim=3, c_stab=0.1, force_stab=False, reinitialize=False, array_convention='C', solver='auto', solve_tol=1e-08, apply_positions=True, apply_cell=True, estimate_mu_eigmode=False, logfile=None, rng=None, neighbour_list=neighbor_list):
        """Initialise a preconditioner object based on passed parameters.

        Parameters:
            r_cut: float. This is a cut-off radius. The preconditioner matrix
                will be created by considering pairs of atoms that are within a
                distance r_cut of each other. For a regular lattice, this is
                usually taken somewhere between the first- and second-nearest
                neighbour distance. If r_cut is not provided, default is
                2 * r_NN (see below)
            r_NN: nearest neighbour distance. If not provided, this is
                  calculated
                from input structure.
            mu: float
                energy scale for position degreees of freedom. If `None`, mu
                is precomputed using finite difference derivatives.
            mu_c: float
                energy scale for cell degreees of freedom. Also precomputed
                if None.
            estimate_mu_eigmode:
                If True, estimates mu based on the lowest eigenmodes of
                unstabilised preconditioner. If False it uses the sine based
                approach.
            dim: int; dimensions of the problem
            c_stab: float. The diagonal of the preconditioner matrix will have
                a stabilisation constant added, which will be the value of
                c_stab times mu.
            force_stab:
                If True, always add the stabilisation to diagnonal, regardless
                of the presence of fixed atoms.
            reinitialize: if True, the value of mu will be recalculated when
                self.make_precon is called. This can be overridden in specific
                cases with reinitialize argument in self.make_precon. If it
                is set to True here, the value passed for mu will be
                irrelevant unless reinitialize is set to False the first time
                make_precon is called.
            array_convention: Either 'C' or 'F' for Fortran; this will change
                the preconditioner to reflect the ordering of the indices in
                the vector it will operate on. The C convention assumes the
                vector will be arranged atom-by-atom (ie [x1, y1, z1, x2, ...])
                while the F convention assumes it will be arranged component
                by component (ie [x1, x2, ..., y1, y2, ...]).
            solver: One of "auto", "direct" or "pyamg", specifying whether to
                use a direct sparse solver or PyAMG to solve P x = y.
                Default is "auto" which uses PyAMG if available, falling
                back to sparse solver if not. solve_tol: tolerance used for
                PyAMG sparse linear solver, if available.
            apply_positions:  bool
                if True, apply preconditioner to position DoF
            apply_cell: bool
                if True, apply preconditioner to cell DoF
            logfile: file object or str
                If *logfile* is a string, a file with that name will be opened.
                Use '-' for stdout.
            rng: None or np.random.RandomState instance
                Random number generator to use for initialising pyamg solver
            neighbor_list: function (optional). Optionally replace the built-in
                ASE neighbour list with an alternative with the same call
                signature, e.g. `matscipy.neighbours.neighbour_list`.

        Raises:
            ValueError for problem with arguments

        """
        self.r_NN = r_NN
        self.r_cut = r_cut
        self.mu = mu
        self.mu_c = mu_c
        self.estimate_mu_eigmode = estimate_mu_eigmode
        self.c_stab = c_stab
        self.force_stab = force_stab
        self.array_convention = array_convention
        self.reinitialize = reinitialize
        self.P = None
        self.old_positions = None
        use_pyamg = False
        if solver == 'auto':
            use_pyamg = have_pyamg
        elif solver == 'direct':
            use_pyamg = False
        elif solver == 'pyamg':
            if not have_pyamg:
                raise RuntimeError("solver='pyamg', PyAMG can't be imported!")
            use_pyamg = True
        else:
            raise ValueError('unknown solver - should be "auto", "direct" or "pyamg"')
        self.use_pyamg = use_pyamg
        self.solve_tol = solve_tol
        self.apply_positions = apply_positions
        self.apply_cell = apply_cell
        if dim < 1:
            raise ValueError('Dimension must be at least 1')
        self.dim = dim
        self.logfile = Logfile(logfile)
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.neighbor_list = neighbor_list

    def copy(self):
        return copy.deepcopy(self)

    def Pdot(self, x):
        return self.P.dot(x)

    def solve(self, x):
        start_time = time.time()
        if self.use_pyamg and have_pyamg:
            y = self.ml.solve(x, x0=self.rng.rand(self.P.shape[0]), tol=self.solve_tol, accel='cg', maxiter=300, cycle='W')
        else:
            y = spsolve(self.P, x)
        self.logfile.write('--- Precon applied in %s seconds ---\n' % (time.time() - start_time))
        return y

    def estimate_mu(self, atoms, H=None):
        """Estimate optimal preconditioner coefficient \\mu

        \\mu is estimated from a numerical solution of

            [dE(p+v) -  dE(p)] \\cdot v = \\mu < P1 v, v >

        with perturbation

            v(x,y,z) = H P_lowest_nonzero_eigvec(x, y, z)

            or

            v(x,y,z) = H (sin(x / Lx), sin(y / Ly), sin(z / Lz))

        After the optimal \\mu is found, self.mu will be set to its value.

        If `atoms` is an instance of Filter an additional \\mu_c
        will be computed for the cell degrees of freedom .

        Args:
            atoms: Atoms object for initial system

            H: 3x3 array or None
                Magnitude of deformation to apply.
                Default is 1e-2*rNN*np.eye(3)

        Returns:
            mu   : float
            mu_c : float or None
        """
        logfile = self.logfile
        if self.dim != 3:
            raise ValueError('Automatic calculation of mu only possible for three-dimensional preconditioners. Try setting mu manually instead.')
        if self.r_NN is None:
            self.r_NN = estimate_nearest_neighbour_distance(atoms, self.neighbor_list)
        if H is None:
            H = 0.01 * self.r_NN * np.eye(3)
        p = atoms.get_positions()
        if self.estimate_mu_eigmode:
            self.mu = 1.0
            self.mu_c = 1.0
            c_stab = self.c_stab
            self.c_stab = 0.0
            if isinstance(atoms, Filter):
                n = len(atoms.atoms)
            else:
                n = len(atoms)
            self._make_sparse_precon(atoms, initial_assembly=True)
            self.P = self.P[:3 * n, :3 * n]
            eigvals, eigvecs = sparse.linalg.eigsh(self.P, k=4, which='SM')
            logfile.write('estimate_mu(): lowest 4 eigvals = %f %f %f %f\n' % (eigvals[0], eigvals[1], eigvals[2], eigvals[3]))
            if any(eigvals[0:3] > 1e-06):
                raise ValueError('First 3 eigenvalues of preconditioner matrixdo not correspond to translational modes.')
            elif eigvals[3] < 1e-06:
                raise ValueError('Fourth smallest eigenvalue of preconditioner matrix is too small, increase r_cut.')
            x = np.zeros(n)
            for i in range(n):
                x[i] = eigvecs[:, 3][3 * i]
            x = x / np.linalg.norm(x)
            if x[0] < 0:
                x = -x
            v = np.zeros(3 * len(atoms))
            for i in range(n):
                v[3 * i] = x[i]
                v[3 * i + 1] = x[i]
                v[3 * i + 2] = x[i]
            v = v / np.linalg.norm(v)
            v = v.reshape((-1, 3))
            self.c_stab = c_stab
        else:
            Lx, Ly, Lz = [p[:, i].max() - p[:, i].min() for i in range(3)]
            logfile.write('estimate_mu(): Lx=%.1f Ly=%.1f Lz=%.1f\n' % (Lx, Ly, Lz))
            x, y, z = p.T
            sine_vr = [x, y, z]
            for i, L in enumerate([Lx, Ly, Lz]):
                if L == 0:
                    warnings.warn('Cell length L[%d] == 0. Setting H[%d,%d] = 0.' % (i, i, i))
                    H[i, i] = 0.0
                else:
                    sine_vr[i] = np.sin(sine_vr[i] / L)
            v = np.dot(H, sine_vr).T
        natoms = len(atoms)
        if isinstance(atoms, Filter):
            natoms = len(atoms.atoms)
            eps = H / self.r_NN
            v[natoms:, :] = eps
        v1 = v.reshape(-1)
        dE_p = -atoms.get_forces().reshape(-1)
        atoms_v = atoms.copy()
        atoms_v.calc = atoms.calc
        if isinstance(atoms, Filter):
            atoms_v = atoms.__class__(atoms_v)
            if hasattr(atoms, 'constant_volume'):
                atoms_v.constant_volume = atoms.constant_volume
        atoms_v.set_positions(p + v)
        dE_p_plus_v = -atoms_v.get_forces().reshape(-1)
        LHS = (dE_p_plus_v - dE_p) * v1
        self.mu = 1.0
        self.mu_c = 1.0
        self._make_sparse_precon(atoms, initial_assembly=True)
        RHS = self.P.dot(v1) * v1
        self.mu = longsum(LHS[:3 * natoms]) / longsum(RHS[:3 * natoms])
        if self.mu < 1.0:
            logfile.write('estimate_mu(): mu (%.3f) < 1.0, capping at mu=1.0' % self.mu)
            self.mu = 1.0
        if isinstance(atoms, Filter):
            self.mu_c = longsum(LHS[3 * natoms:]) / longsum(RHS[3 * natoms:])
            if self.mu_c < 1.0:
                logfile.write('estimate_mu(): mu_c (%.3f) < 1.0, capping at mu_c=1.0\n' % self.mu_c)
                self.mu_c = 1.0
        logfile.write('estimate_mu(): mu=%r, mu_c=%r\n' % (self.mu, self.mu_c))
        self.P = None
        return (self.mu, self.mu_c)

    def asarray(self):
        return np.array(self.P.todense())

    def one_dim_to_ndim(self, csc_P, N):
        """
        Expand an N x N precon matrix to self.dim*N x self.dim * N

        Args:
            csc_P (sparse matrix): N x N sparse matrix, in CSC format
        """
        start_time = time.time()
        if self.dim == 1:
            P = csc_P
        elif self.array_convention == 'F':
            csc_P = csc_P.tocsr()
            P = csc_P
            for i in range(self.dim - 1):
                P = sparse.block_diag((P, csc_P)).tocsr()
        else:
            csc_P = csc_P.tocoo()
            i = csc_P.row * self.dim
            j = csc_P.col * self.dim
            z = csc_P.data
            I = np.hstack([i + d for d in range(self.dim)])
            J = np.hstack([j + d for d in range(self.dim)])
            Z = np.hstack([z for d in range(self.dim)])
            P = sparse.csc_matrix((Z, (I, J)), shape=(self.dim * N, self.dim * N))
            P = P.tocsr()
        self.logfile.write('--- N-dim precon created in %s s ---\n' % (time.time() - start_time))
        return P

    def create_solver(self):
        if self.use_pyamg and have_pyamg:
            start_time = time.time()
            self.ml = create_pyamg_solver(self.P)
            self.logfile.write('--- multi grid solver created in %s ---\n' % (time.time() - start_time))