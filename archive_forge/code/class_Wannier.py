from time import time
from math import sqrt, pi
import numpy as np
from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize
from ase.io.jsonio import read_json, write_json
class Wannier:
    """Maximally localized Wannier Functions

    Find the set of maximally localized Wannier functions using the
    spread functional of Marzari and Vanderbilt (PRB 56, 1997 page
    12847).
    """

    def __init__(self, nwannier, calc, file=None, nbands=None, fixedenergy=None, fixedstates=None, spin=0, initialwannier='random', rng=np.random, verbose=False):
        """
        Required arguments:

          ``nwannier``: The number of Wannier functions you wish to construct.
            This must be at least half the number of electrons in the system
            and at most equal to the number of bands in the calculation.

          ``calc``: A converged DFT calculator class.
            If ``file`` arg. is not provided, the calculator *must* provide the
            method ``get_wannier_localization_matrix``, and contain the
            wavefunctions (save files with only the density is not enough).
            If the localization matrix is read from file, this is not needed,
            unless ``get_function`` or ``write_cube`` is called.

        Optional arguments:

          ``nbands``: Bands to include in localization.
            The number of bands considered by Wannier can be smaller than the
            number of bands in the calculator. This is useful if the highest
            bands of the DFT calculation are not well converged.

          ``spin``: The spin channel to be considered.
            The Wannier code treats each spin channel independently.

          ``fixedenergy`` / ``fixedstates``: Fixed part of Heilbert space.
            Determine the fixed part of Hilbert space by either a maximal
            energy *or* a number of bands (possibly a list for multiple
            k-points).
            Default is None meaning that the number of fixed states is equated
            to ``nwannier``.

          ``file``: Read localization and rotation matrices from this file.

          ``initialwannier``: Initial guess for Wannier rotation matrix.
            Can be 'bloch' to start from the Bloch states, 'random' to be
            randomized, or a list passed to calc.get_initial_wannier.

          ``rng``: Random number generator for ``initialwannier``.

          ``verbose``: True / False level of verbosity.
          """
        sign = -1
        self.nwannier = nwannier
        self.calc = calc
        self.spin = spin
        self.verbose = verbose
        self.kpt_kc = calc.get_bz_k_points()
        assert len(calc.get_ibz_k_points()) == len(self.kpt_kc)
        self.kptgrid = get_monkhorst_pack_size_and_offset(self.kpt_kc)[0]
        self.kpt_kc *= sign
        self.Nk = len(self.kpt_kc)
        self.unitcell_cc = calc.get_atoms().get_cell()
        self.largeunitcell_cc = (self.unitcell_cc.T * self.kptgrid).T
        self.weight_d, self.Gdir_dc = calculate_weights(self.largeunitcell_cc)
        self.Ndir = len(self.weight_d)
        if nbands is not None:
            self.nbands = nbands
        else:
            self.nbands = calc.get_number_of_bands()
        if fixedenergy is None:
            if fixedstates is None:
                self.fixedstates_k = np.array([nwannier] * self.Nk, int)
            else:
                if isinstance(fixedstates, int):
                    fixedstates = [fixedstates] * self.Nk
                self.fixedstates_k = np.array(fixedstates, int)
        else:
            fixedenergy += calc.get_fermi_level()
            print(fixedenergy)
            self.fixedstates_k = np.array([calc.get_eigenvalues(k, spin).searchsorted(fixedenergy) for k in range(self.Nk)], int)
        self.edf_k = self.nwannier - self.fixedstates_k
        if verbose:
            print('Wannier: Fixed states            : %s' % self.fixedstates_k)
            print('Wannier: Extra degrees of freedom: %s' % self.edf_k)
        if self.Nk == 1:
            self.kklst_dk = np.zeros((self.Ndir, 1), int)
            k0_dkc = self.Gdir_dc.reshape(-1, 1, 3)
        else:
            self.kklst_dk = np.empty((self.Ndir, self.Nk), int)
            k0_dkc = np.empty((self.Ndir, self.Nk, 3), int)
            kdist_c = np.empty(3)
            for c in range(3):
                slist = np.argsort(self.kpt_kc[:, c], kind='mergesort')
                skpoints_kc = np.take(self.kpt_kc, slist, axis=0)
                kdist_c[c] = max([skpoints_kc[n + 1, c] - skpoints_kc[n, c] for n in range(self.Nk - 1)])
            for d, Gdir_c in enumerate(self.Gdir_dc):
                for k, k_c in enumerate(self.kpt_kc):
                    G_c = np.where(Gdir_c > 0, kdist_c, 0)
                    if max(G_c) < 0.0001:
                        self.kklst_dk[d, k] = k
                        k0_dkc[d, k] = Gdir_c
                    else:
                        self.kklst_dk[d, k], k0_dkc[d, k] = neighbor_k_search(k_c, G_c, self.kpt_kc)
        self.invkklst_dk = np.empty((self.Ndir, self.Nk), int)
        for d in range(self.Ndir):
            for k1 in range(self.Nk):
                self.invkklst_dk[d, k1] = self.kklst_dk[d].tolist().index(k1)
        Nw = self.nwannier
        Nb = self.nbands
        self.Z_dkww = np.empty((self.Ndir, self.Nk, Nw, Nw), complex)
        self.V_knw = np.zeros((self.Nk, Nb, Nw), complex)
        if file is None:
            self.Z_dknn = np.empty((self.Ndir, self.Nk, Nb, Nb), complex)
            for d, dirG in enumerate(self.Gdir_dc):
                for k in range(self.Nk):
                    k1 = self.kklst_dk[d, k]
                    k0_c = k0_dkc[d, k]
                    self.Z_dknn[d, k] = calc.get_wannier_localization_matrix(nbands=Nb, dirG=dirG, kpoint=k, nextkpoint=k1, G_I=k0_c, spin=self.spin)
        self.initialize(file=file, initialwannier=initialwannier, rng=rng)

    def initialize(self, file=None, initialwannier='random', rng=np.random):
        """Re-initialize current rotation matrix.

        Keywords are identical to those of the constructor.
        """
        Nw = self.nwannier
        Nb = self.nbands
        if file is not None:
            with paropen(file, 'r') as fd:
                self.Z_dknn, self.U_kww, self.C_kul = read_json(fd)
        elif initialwannier == 'bloch':
            self.U_kww = np.zeros((self.Nk, Nw, Nw), complex)
            self.C_kul = []
            for U, M, L in zip(self.U_kww, self.fixedstates_k, self.edf_k):
                U[:] = np.identity(Nw, complex)
                if L > 0:
                    self.C_kul.append(np.identity(Nb - M, complex)[:, :L])
                else:
                    self.C_kul.append([])
        elif initialwannier == 'random':
            self.U_kww = np.zeros((self.Nk, Nw, Nw), complex)
            self.C_kul = []
            for U, M, L in zip(self.U_kww, self.fixedstates_k, self.edf_k):
                U[:] = random_orthogonal_matrix(Nw, rng, real=False)
                if L > 0:
                    self.C_kul.append(random_orthogonal_matrix(Nb - M, rng=rng, real=False)[:, :L])
                else:
                    self.C_kul.append(np.array([]))
        else:
            self.C_kul, self.U_kww = self.calc.initial_wannier(initialwannier, self.kptgrid, self.fixedstates_k, self.edf_k, self.spin, self.nbands)
        self.update()

    def save(self, file):
        """Save information on localization and rotation matrices to file."""
        with paropen(file, 'w') as fd:
            write_json(fd, (self.Z_dknn, self.U_kww, self.C_kul))

    def update(self):
        for k, M in enumerate(self.fixedstates_k):
            self.V_knw[k, :M] = self.U_kww[k, :M]
            if M < self.nwannier:
                self.V_knw[k, M:] = np.dot(self.C_kul[k], self.U_kww[k, M:])
        for d in range(self.Ndir):
            for k in range(self.Nk):
                k1 = self.kklst_dk[d, k]
                self.Z_dkww[d, k] = np.dot(dag(self.V_knw[k]), np.dot(self.Z_dknn[d, k], self.V_knw[k1]))
        self.Z_dww = self.Z_dkww.sum(axis=1) / self.Nk

    def get_centers(self, scaled=False):
        """Calculate the Wannier centers

        ::

          pos =  L / 2pi * phase(diag(Z))
        """
        coord_wc = np.angle(self.Z_dww[:3].diagonal(0, 1, 2)).T / (2 * pi) % 1
        if not scaled:
            coord_wc = np.dot(coord_wc, self.largeunitcell_cc)
        return coord_wc

    def get_radii(self):
        """Calculate the spread of the Wannier functions.

        ::

                        --  /  L  \\ 2       2
          radius**2 = - >   | --- |   ln |Z|
                        --d \\ 2pi /
        """
        r2 = -np.dot(self.largeunitcell_cc.diagonal() ** 2 / (2 * pi) ** 2, np.log(abs(self.Z_dww[:3].diagonal(0, 1, 2)) ** 2))
        return np.sqrt(r2)

    def get_spectral_weight(self, w):
        return abs(self.V_knw[:, :, w]) ** 2 / self.Nk

    def get_pdos(self, w, energies, width):
        """Projected density of states (PDOS).

        Returns the (PDOS) for Wannier function ``w``. The calculation
        is performed over the energy grid specified in energies. The
        PDOS is produced as a sum of Gaussians centered at the points
        of the energy grid and with the specified width.
        """
        spec_kn = self.get_spectral_weight(w)
        dos = np.zeros(len(energies))
        for k, spec_n in enumerate(spec_kn):
            eig_n = self.calc.get_eigenvalues(kpt=k, spin=self.spin)
            for weight, eig in zip(spec_n, eig_n):
                x = ((energies - eig) / width) ** 2
                dos += weight * np.exp(-x.clip(0.0, 40.0)) / (sqrt(pi) * width)
        return dos

    def translate(self, w, R):
        """Translate the w'th Wannier function

        The distance vector R = [n1, n2, n3], is in units of the basis
        vectors of the small cell.
        """
        for kpt_c, U_ww in zip(self.kpt_kc, self.U_kww):
            U_ww[:, w] *= np.exp(2j * pi * np.dot(np.array(R), kpt_c))
        self.update()

    def translate_to_cell(self, w, cell):
        """Translate the w'th Wannier function to specified cell"""
        scaled_c = np.angle(self.Z_dww[:3, w, w]) * self.kptgrid / (2 * pi)
        trans = np.array(cell) - np.floor(scaled_c)
        self.translate(w, trans)

    def translate_all_to_cell(self, cell=[0, 0, 0]):
        """Translate all Wannier functions to specified cell.

        Move all Wannier orbitals to a specific unit cell.  There
        exists an arbitrariness in the positions of the Wannier
        orbitals relative to the unit cell. This method can move all
        orbitals to the unit cell specified by ``cell``.  For a
        `\\Gamma`-point calculation, this has no effect. For a
        **k**-point calculation the periodicity of the orbitals are
        given by the large unit cell defined by repeating the original
        unitcell by the number of **k**-points in each direction.  In
        this case it is useful to move the orbitals away from the
        boundaries of the large cell before plotting them. For a bulk
        calculation with, say 10x10x10 **k** points, one could move
        the orbitals to the cell [2,2,2].  In this way the pbc
        boundary conditions will not be noticed.
        """
        scaled_wc = np.angle(self.Z_dww[:3].diagonal(0, 1, 2)).T * self.kptgrid / (2 * pi)
        trans_wc = np.array(cell)[None] - np.floor(scaled_wc)
        for kpt_c, U_ww in zip(self.kpt_kc, self.U_kww):
            U_ww *= np.exp(2j * pi * np.dot(trans_wc, kpt_c))
        self.update()

    def distances(self, R):
        """Relative distances between centers.

        Returns a matrix with the distances between different Wannier centers.
        R = [n1, n2, n3] is in units of the basis vectors of the small cell
        and allows one to measure the distance with centers moved to a
        different small cell.
        The dimension of the matrix is [Nw, Nw].
        """
        Nw = self.nwannier
        cen = self.get_centers()
        r1 = cen.repeat(Nw, axis=0).reshape(Nw, Nw, 3)
        r2 = cen.copy()
        for i in range(3):
            r2 += self.unitcell_cc[i] * R[i]
        r2 = np.swapaxes(r2.repeat(Nw, axis=0).reshape(Nw, Nw, 3), 0, 1)
        return np.sqrt(np.sum((r1 - r2) ** 2, axis=-1))

    def get_hopping(self, R):
        """Returns the matrix H(R)_nm=<0,n|H|R,m>.

        ::

                                1   _   -ik.R
          H(R) = <0,n|H|R,m> = --- >_  e      H(k)
                                Nk  k

        where R is the cell-distance (in units of the basis vectors of
        the small cell) and n,m are indices of the Wannier functions.
        """
        H_ww = np.zeros([self.nwannier, self.nwannier], complex)
        for k, kpt_c in enumerate(self.kpt_kc):
            phase = np.exp(-2j * pi * np.dot(np.array(R), kpt_c))
            H_ww += self.get_hamiltonian(k) * phase
        return H_ww / self.Nk

    def get_hamiltonian(self, k=0):
        """Get Hamiltonian at existing k-vector of index k

        ::

                  dag
          H(k) = V    diag(eps )  V
                  k           k    k
        """
        eps_n = self.calc.get_eigenvalues(kpt=k, spin=self.spin)[:self.nbands]
        return np.dot(dag(self.V_knw[k]) * eps_n, self.V_knw[k])

    def get_hamiltonian_kpoint(self, kpt_c):
        """Get Hamiltonian at some new arbitrary k-vector

        ::

                  _   ik.R
          H(k) = >_  e     H(R)
                  R

        Warning: This method moves all Wannier functions to cell (0, 0, 0)
        """
        if self.verbose:
            print('Translating all Wannier functions to cell (0, 0, 0)')
        self.translate_all_to_cell()
        max = (self.kptgrid - 1) // 2
        N1, N2, N3 = max
        Hk = np.zeros([self.nwannier, self.nwannier], complex)
        for n1 in range(-N1, N1 + 1):
            for n2 in range(-N2, N2 + 1):
                for n3 in range(-N3, N3 + 1):
                    R = np.array([n1, n2, n3], float)
                    hop_ww = self.get_hopping(R)
                    phase = np.exp(+2j * pi * np.dot(R, kpt_c))
                    Hk += hop_ww * phase
        return Hk

    def get_function(self, index, repeat=None):
        """Get Wannier function on grid.

        Returns an array with the funcion values of the indicated Wannier
        function on a grid with the size of the *repeated* unit cell.

        For a calculation using **k**-points the relevant unit cell for
        eg. visualization of the Wannier orbitals is not the original unit
        cell, but rather a larger unit cell defined by repeating the
        original unit cell by the number of **k**-points in each direction.
        Note that for a `\\Gamma`-point calculation the large unit cell
        coinsides with the original unit cell.
        The large unitcell also defines the periodicity of the Wannier
        orbitals.

        ``index`` can be either a single WF or a coordinate vector in terms
        of the WFs.
        """
        if repeat is None:
            repeat = self.kptgrid
        N1, N2, N3 = repeat
        dim = self.calc.get_number_of_grid_points()
        largedim = dim * [N1, N2, N3]
        wanniergrid = np.zeros(largedim, dtype=complex)
        for k, kpt_c in enumerate(self.kpt_kc):
            if isinstance(index, int):
                vec_n = self.V_knw[k, :, index]
            else:
                vec_n = np.dot(self.V_knw[k], index)
            wan_G = np.zeros(dim, complex)
            for n, coeff in enumerate(vec_n):
                wan_G += coeff * self.calc.get_pseudo_wave_function(n, k, self.spin, pad=True)
            for n1 in range(N1):
                for n2 in range(N2):
                    for n3 in range(N3):
                        e = np.exp(-2j * pi * np.dot([n1, n2, n3], kpt_c))
                        wanniergrid[n1 * dim[0]:(n1 + 1) * dim[0], n2 * dim[1]:(n2 + 1) * dim[1], n3 * dim[2]:(n3 + 1) * dim[2]] += e * wan_G
        wanniergrid /= np.sqrt(self.Nk)
        return wanniergrid

    def write_cube(self, index, fname, repeat=None, real=True):
        """Dump specified Wannier function to a cube file"""
        from ase.io import write
        if repeat is None:
            repeat = self.kptgrid
        atoms = self.calc.get_atoms() * repeat
        func = self.get_function(index, repeat)
        if real:
            if self.Nk == 1:
                func *= np.exp(-1j * np.angle(func.max()))
                if 0:
                    assert max(abs(func.imag).flat) < 0.0001
                func = func.real
            else:
                func = abs(func)
        else:
            phase_fname = fname.split('.')
            phase_fname.insert(1, 'phase')
            phase_fname = '.'.join(phase_fname)
            write(phase_fname, atoms, data=np.angle(func), format='cube')
            func = abs(func)
        write(fname, atoms, data=func, format='cube')

    def localize(self, step=0.25, tolerance=1e-08, updaterot=True, updatecoeff=True):
        """Optimize rotation to give maximal localization"""
        md_min(self, step, tolerance, verbose=self.verbose, updaterot=updaterot, updatecoeff=updatecoeff)

    def get_functional_value(self):
        """Calculate the value of the spread functional.

        ::

          Tr[|ZI|^2]=sum(I)sum(n) w_i|Z_(i)_nn|^2,

        where w_i are weights."""
        a_d = np.sum(np.abs(self.Z_dww.diagonal(0, 1, 2)) ** 2, axis=1)
        return np.dot(a_d, self.weight_d).real

    def get_gradients(self):
        Nb = self.nbands
        Nw = self.nwannier
        dU = []
        dC = []
        for k in range(self.Nk):
            M = self.fixedstates_k[k]
            L = self.edf_k[k]
            U_ww = self.U_kww[k]
            C_ul = self.C_kul[k]
            Utemp_ww = np.zeros((Nw, Nw), complex)
            Ctemp_nw = np.zeros((Nb, Nw), complex)
            for d, weight in enumerate(self.weight_d):
                if abs(weight) < 1e-06:
                    continue
                Z_knn = self.Z_dknn[d]
                diagZ_w = self.Z_dww[d].diagonal()
                Zii_ww = np.repeat(diagZ_w, Nw).reshape(Nw, Nw)
                k1 = self.kklst_dk[d, k]
                k2 = self.invkklst_dk[d, k]
                V_knw = self.V_knw
                Z_kww = self.Z_dkww[d]
                if L > 0:
                    Ctemp_nw += weight * np.dot(np.dot(Z_knn[k], V_knw[k1]) * diagZ_w.conj() + np.dot(dag(Z_knn[k2]), V_knw[k2]) * diagZ_w, dag(U_ww))
                temp = Zii_ww.T * Z_kww[k].conj() - Zii_ww * Z_kww[k2].conj()
                Utemp_ww += weight * (temp - dag(temp))
            dU.append(Utemp_ww.ravel())
            if L > 0:
                Ctemp_ul = Ctemp_nw[M:, M:]
                G_ul = Ctemp_ul - np.dot(np.dot(C_ul, dag(C_ul)), Ctemp_ul)
                dC.append(G_ul.ravel())
        return np.concatenate(dU + dC)

    def step(self, dX, updaterot=True, updatecoeff=True):
        Nw = self.nwannier
        Nk = self.Nk
        M_k = self.fixedstates_k
        L_k = self.edf_k
        if updaterot:
            A_kww = dX[:Nk * Nw ** 2].reshape(Nk, Nw, Nw)
            for U, A in zip(self.U_kww, A_kww):
                H = -1j * A.conj()
                epsilon, Z = np.linalg.eigh(H)
                dU = np.dot(Z * np.exp(1j * epsilon), dag(Z))
                if U.dtype == float:
                    U[:] = np.dot(U, dU).real
                else:
                    U[:] = np.dot(U, dU)
        if updatecoeff:
            start = 0
            for C, unocc, L in zip(self.C_kul, self.nbands - M_k, L_k):
                if L == 0 or unocc == 0:
                    continue
                Ncoeff = L * unocc
                deltaC = dX[Nk * Nw ** 2 + start:Nk * Nw ** 2 + start + Ncoeff]
                C += deltaC.reshape(unocc, L)
                gram_schmidt(C)
                start += Ncoeff
        self.update()