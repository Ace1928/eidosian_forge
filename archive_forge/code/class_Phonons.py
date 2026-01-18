from math import pi, sqrt
import warnings
from pathlib import Path
import numpy as np
import numpy.linalg as la
import numpy.fft as fft
import ase
import ase.units as units
from ase.parallel import world
from ase.dft import monkhorst_pack
from ase.io.trajectory import Trajectory
from ase.utils.filecache import MultiFileJSONCache
class Phonons(Displacement):
    """Class for calculating phonon modes using the finite displacement method.

    The matrix of force constants is calculated from the finite difference
    approximation to the first-order derivative of the atomic forces as::

                            2             nbj   nbj
                nbj        d E           F-  - F+
               C     = ------------ ~  -------------  ,
                mai     dR   dR          2 * delta
                          mai  nbj

    where F+/F- denotes the force in direction j on atom nb when atom ma is
    displaced in direction +i/-i. The force constants are related by various
    symmetry relations. From the definition of the force constants it must
    be symmetric in the three indices mai::

                nbj    mai         bj        ai
               C    = C      ->   C  (R ) = C  (-R )  .
                mai    nbj         ai  n     bj   n

    As the force constants can only depend on the difference between the m and
    n indices, this symmetry is more conveniently expressed as shown on the
    right hand-side.

    The acoustic sum-rule::

                           _ _
                aj         \\    bj
               C  (R ) = -  )  C  (R )
                ai  0      /__  ai  m
                          (m, b)
                            !=
                          (0, a)

    Ordering of the unit cells illustrated here for a 1-dimensional system (in
    case ``refcell=None`` in constructor!):

    ::

               m = 0        m = 1        m = -2        m = -1
           -----------------------------------------------------
           |            |            |            |            |
           |        * b |        *   |        *   |        *   |
           |            |            |            |            |
           |   * a      |   *        |   *        |   *        |
           |            |            |            |            |
           -----------------------------------------------------

    Example:

    >>> from ase.build import bulk
    >>> from ase.phonons import Phonons
    >>> from gpaw import GPAW, FermiDirac
    >>> atoms = bulk('Si', 'diamond', a=5.4)
    >>> calc = GPAW(kpts=(5, 5, 5),
                    h=0.2,
                    occupations=FermiDirac(0.))
    >>> ph = Phonons(atoms, calc, supercell=(5, 5, 5))
    >>> ph.run()
    >>> ph.read(method='frederiksen', acoustic=True)

    """

    def __init__(self, *args, **kwargs):
        """Initialize with base class args and kwargs."""
        if 'name' not in kwargs:
            kwargs['name'] = 'phonon'
        self.deprecate_refcell(kwargs)
        Displacement.__init__(self, *args, **kwargs)
        self.C_N = None
        self.D_N = None
        self.Z_avv = None
        self.eps_vv = None

    @staticmethod
    def deprecate_refcell(kwargs: dict):
        if 'refcell' in kwargs:
            warnings.warn('Keyword refcell of Phonons is deprecated.Please use center_refcell (bool)', FutureWarning)
            kwargs['center_refcell'] = bool(kwargs['refcell'])
            kwargs.pop('refcell')
        return kwargs

    def __call__(self, atoms_N):
        """Calculate forces on atoms in supercell."""
        return atoms_N.get_forces()

    def calculate(self, atoms_N, disp):
        forces = self(atoms_N)
        return {'forces': forces}

    def check_eq_forces(self):
        """Check maximum size of forces in the equilibrium structure."""
        name = f'{self.name}.eq'
        feq_av = self.cache[name]['forces']
        fmin = feq_av.max()
        fmax = feq_av.min()
        i_min = np.where(feq_av == fmin)
        i_max = np.where(feq_av == fmax)
        return (fmin, fmax, i_min, i_max)

    def read_born_charges(self, name=None, neutrality=True):
        """Read Born charges and dieletric tensor from JSON file.

        The charge neutrality sum-rule::

                   _ _
                   \\    a
                    )  Z   = 0
                   /__  ij
                    a

        Parameters:

        neutrality: bool
            Restore charge neutrality condition on calculated Born effective
            charges.

        """
        if name is None:
            key = '%s.born' % self.name
        else:
            key = name
        Z_avv, eps_vv = self.cache[key]
        if neutrality:
            Z_mean = Z_avv.sum(0) / len(Z_avv)
            Z_avv -= Z_mean
        self.Z_avv = Z_avv[self.indices]
        self.eps_vv = eps_vv

    def read(self, method='Frederiksen', symmetrize=3, acoustic=True, cutoff=None, born=False, **kwargs):
        """Read forces from json files and calculate force constants.

        Extra keyword arguments will be passed to ``read_born_charges``.

        Parameters:

        method: str
            Specify method for evaluating the atomic forces.
        symmetrize: int
            Symmetrize force constants (see doc string at top) when
            ``symmetrize != 0`` (default: 3). Since restoring the acoustic sum
            rule breaks the symmetry, the symmetrization must be repeated a few
            times until the changes a insignificant. The integer gives the
            number of iterations that will be carried out.
        acoustic: bool
            Restore the acoustic sum rule on the force constants.
        cutoff: None or float
            Zero elements in the dynamical matrix between atoms with an
            interatomic distance larger than the cutoff.
        born: bool
            Read in Born effective charge tensor and high-frequency static
            dielelctric tensor from file.

        """
        method = method.lower()
        assert method in ['standard', 'frederiksen']
        if cutoff is not None:
            cutoff = float(cutoff)
        if born:
            self.read_born_charges(**kwargs)
        natoms = len(self.indices)
        N = np.prod(self.supercell)
        C_xNav = np.empty((natoms * 3, N, natoms, 3), dtype=float)
        for i, a in enumerate(self.indices):
            for j, v in enumerate('xyz'):
                basename = '%d%s' % (a, v)
                fminus_av = self.cache[basename + '-']['forces']
                fplus_av = self.cache[basename + '+']['forces']
                if method == 'frederiksen':
                    fminus_av[a] -= fminus_av.sum(0)
                    fplus_av[a] -= fplus_av.sum(0)
                C_av = fminus_av - fplus_av
                C_av /= 2 * self.delta
                C_Nav = C_av.reshape((N, len(self.atoms), 3))[:, self.indices]
                index = 3 * i + j
                C_xNav[index] = C_Nav
        C_N = C_xNav.swapaxes(0, 1).reshape((N,) + (3 * natoms, 3 * natoms))
        if cutoff is not None:
            self.apply_cutoff(C_N, cutoff)
        if symmetrize:
            for i in range(symmetrize):
                C_N = self.symmetrize(C_N)
                if acoustic:
                    self.acoustic(C_N)
                else:
                    break
        self.C_N = C_N
        self.D_N = C_N.copy()
        m_a = self.atoms.get_masses()
        self.m_inv_x = np.repeat(m_a[self.indices] ** (-0.5), 3)
        M_inv = np.outer(self.m_inv_x, self.m_inv_x)
        for D in self.D_N:
            D *= M_inv

    def symmetrize(self, C_N):
        """Symmetrize force constant matrix."""
        natoms = len(self.indices)
        N = np.prod(self.supercell)
        C_lmn = C_N.reshape(self.supercell + (3 * natoms, 3 * natoms))
        if self.offset == 0:
            C_lmn = fft.fftshift(C_lmn, axes=(0, 1, 2)).copy()
        i, j, k = 1 - np.asarray(self.supercell) % 2
        C_lmn[i:, j:, k:] *= 0.5
        C_lmn[i:, j:, k:] += C_lmn[i:, j:, k:][::-1, ::-1, ::-1].transpose(0, 1, 2, 4, 3).copy()
        if self.offset == 0:
            C_lmn = fft.ifftshift(C_lmn, axes=(0, 1, 2)).copy()
        C_N = C_lmn.reshape((N, 3 * natoms, 3 * natoms))
        return C_N

    def acoustic(self, C_N):
        """Restore acoustic sumrule on force constants."""
        natoms = len(self.indices)
        C_N_temp = C_N.copy()
        for C in C_N_temp:
            for a in range(natoms):
                for a_ in range(natoms):
                    C_N[self.offset, 3 * a:3 * a + 3, 3 * a:3 * a + 3] -= C[3 * a:3 * a + 3, 3 * a_:3 * a_ + 3]

    def apply_cutoff(self, D_N, r_c):
        """Zero elements for interatomic distances larger than the cutoff.

        Parameters:

        D_N: ndarray
            Dynamical/force constant matrix.
        r_c: float
            Cutoff in Angstrom.

        """
        natoms = len(self.indices)
        N = np.prod(self.supercell)
        R_cN = self._lattice_vectors_array
        D_Navav = D_N.reshape((N, natoms, 3, natoms, 3))
        cell_vc = self.atoms.cell.transpose()
        pos_av = self.atoms.get_positions()
        for n in range(N):
            R_v = np.dot(cell_vc, R_cN[:, n])
            posn_av = pos_av + R_v
            for i, a in enumerate(self.indices):
                dist_a = np.sqrt(np.sum((pos_av[a] - posn_av) ** 2, axis=-1))
                i_a = dist_a > r_c
                D_Navav[n, i, :, i_a, :] = 0.0

    def get_force_constant(self):
        """Return matrix of force constants."""
        assert self.C_N is not None
        return self.C_N

    def get_band_structure(self, path, modes=False, born=False, verbose=True):
        omega_kl = self.band_structure(path.kpts, modes, born, verbose)
        if modes:
            assert 0
            omega_kl, modes = omega_kl
        from ase.spectrum.band_structure import BandStructure
        bs = BandStructure(path, energies=omega_kl[None])
        return bs

    def compute_dynamical_matrix(self, q_scaled: np.ndarray, D_N: np.ndarray):
        """ Computation of the dynamical matrix in momentum space D_ab(q).
            This is a Fourier transform from real-space dynamical matrix D_N
            for a given momentum vector q.

        q_scaled: q vector in scaled coordinates.

        D_N: the dynamical matrix in real-space. It is necessary, at least
             currently, to provide this matrix explicitly (rather than use
             self.D_N) because this matrix is modified by the Born charges
             contributions and these modifications are momentum (q) dependent.

        Result:
            D(q): two-dimensional, complex-valued array of
                  shape=(3 * natoms, 3 * natoms).
        """
        R_cN = self._lattice_vectors_array
        phase_N = np.exp(-2j * pi * np.dot(q_scaled, R_cN))
        D_q = np.sum(phase_N[:, np.newaxis, np.newaxis] * D_N, axis=0)
        return D_q

    def band_structure(self, path_kc, modes=False, born=False, verbose=True):
        """Calculate phonon dispersion along a path in the Brillouin zone.

        The dynamical matrix at arbitrary q-vectors is obtained by Fourier
        transforming the real-space force constants. In case of negative
        eigenvalues (squared frequency), the corresponding negative frequency
        is returned.

        Frequencies and modes are in units of eV and Ang/sqrt(amu),
        respectively.

        Parameters:

        path_kc: ndarray
            List of k-point coordinates (in units of the reciprocal lattice
            vectors) specifying the path in the Brillouin zone for which the
            dynamical matrix will be calculated.
        modes: bool
            Returns both frequencies and modes when True.
        born: bool
            Include non-analytic part given by the Born effective charges and
            the static part of the high-frequency dielectric tensor. This
            contribution to the force constant accounts for the splitting
            between the LO and TO branches for q -> 0.
        verbose: bool
            Print warnings when imaginary frequncies are detected.

        """
        assert self.D_N is not None
        if born:
            assert self.Z_avv is not None
            assert self.eps_vv is not None
        D_N = self.D_N
        omega_kl = []
        u_kl = []
        reci_vc = 2 * pi * la.inv(self.atoms.cell)
        vol = abs(la.det(self.atoms.cell)) / units.Bohr ** 3
        for q_c in path_kc:
            if born:
                q_v = np.dot(reci_vc, q_c)
                qdotZ_av = np.dot(q_v, self.Z_avv).ravel()
                C_na = 4 * pi * np.outer(qdotZ_av, qdotZ_av) / np.dot(q_v, np.dot(self.eps_vv, q_v)) / vol
                self.C_na = C_na / units.Bohr ** 2 * units.Hartree
                M_inv = np.outer(self.m_inv_x, self.m_inv_x)
                D_na = C_na * M_inv / units.Bohr ** 2 * units.Hartree
                self.D_na = D_na
                D_N = self.D_N + D_na / np.prod(self.supercell)
            D_q = self.compute_dynamical_matrix(q_c, D_N)
            if modes:
                omega2_l, u_xl = la.eigh(D_q, UPLO='U')
                u_lx = (self.m_inv_x[:, np.newaxis] * u_xl[:, omega2_l.argsort()]).T.copy()
                u_kl.append(u_lx.reshape((-1, len(self.indices), 3)))
            else:
                omega2_l = la.eigvalsh(D_q, UPLO='U')
            omega2_l.sort()
            omega_l = np.sqrt(omega2_l.astype(complex))
            if not np.all(omega2_l >= 0.0):
                indices = np.where(omega2_l < 0)[0]
                if verbose:
                    print('WARNING, %i imaginary frequencies at q = (% 5.2f, % 5.2f, % 5.2f) ; (omega_q =% 5.3e*i)' % (len(indices), q_c[0], q_c[1], q_c[2], omega_l[indices][0].imag))
                omega_l[indices] = -1 * np.sqrt(np.abs(omega2_l[indices].real))
            omega_kl.append(omega_l.real)
        s = units._hbar * 10000000000.0 / sqrt(units._e * units._amu)
        omega_kl = s * np.asarray(omega_kl)
        if modes:
            return (omega_kl, np.asarray(u_kl))
        return omega_kl

    def get_dos(self, kpts=(10, 10, 10), npts=1000, delta=0.001, indices=None):
        from ase.spectrum.dosdata import RawDOSData
        kpts_kc = monkhorst_pack(kpts)
        omega_w = self.band_structure(kpts_kc).ravel()
        dos = RawDOSData(omega_w, np.ones_like(omega_w))
        return dos

    def dos(self, kpts=(10, 10, 10), npts=1000, delta=0.001, indices=None):
        """Calculate phonon dos as a function of energy.

        Parameters:

        qpts: tuple
            Shape of Monkhorst-Pack grid for sampling the Brillouin zone.
        npts: int
            Number of energy points.
        delta: float
            Broadening of Lorentzian line-shape in eV.
        indices: list
            If indices is not None, the atomic-partial dos for the specified
            atoms will be calculated.

        """
        kpts_kc = monkhorst_pack(kpts)
        N = np.prod(kpts)
        omega_kl = self.band_structure(kpts_kc)
        omega_e = np.linspace(0.0, np.amax(omega_kl) + 0.005, num=npts)
        dos_e = np.zeros_like(omega_e)
        for omega_l in omega_kl:
            diff_el = (omega_e[:, np.newaxis] - omega_l[np.newaxis, :]) ** 2
            dos_el = 1.0 / (diff_el + (0.5 * delta) ** 2)
            dos_e += dos_el.sum(axis=1)
        dos_e *= 1.0 / (N * pi) * 0.5 * delta
        return (omega_e, dos_e)

    def write_modes(self, q_c, branches=0, kT=units.kB * 300, born=False, repeat=(1, 1, 1), nimages=30, center=False):
        """Write modes to trajectory file.

        Parameters:

        q_c: ndarray
            q-vector of the modes.
        branches: int or list
            Branch index of modes.
        kT: float
            Temperature in units of eV. Determines the amplitude of the atomic
            displacements in the modes.
        born: bool
            Include non-analytic contribution to the force constants at q -> 0.
        repeat: tuple
            Repeat atoms (l, m, n) times in the directions of the lattice
            vectors. Displacements of atoms in repeated cells carry a Bloch
            phase factor given by the q-vector and the cell lattice vector R_m.
        nimages: int
            Number of images in an oscillation.
        center: bool
            Center atoms in unit cell if True (default: False).

        """
        if isinstance(branches, int):
            branch_l = [branches]
        else:
            branch_l = list(branches)
        omega_l, u_l = self.band_structure([q_c], modes=True, born=born)
        atoms = self.atoms * repeat
        if center:
            atoms.center()
        pos_Nav = atoms.get_positions()
        N = np.prod(repeat)
        R_cN = np.indices(repeat).reshape(3, -1)
        phase_N = np.exp(2j * pi * np.dot(q_c, R_cN))
        phase_Na = phase_N.repeat(len(self.atoms))
        for l in branch_l:
            omega = omega_l[0, l]
            u_av = u_l[0, l]
            u_av *= sqrt(kT) / abs(omega)
            mode_av = np.zeros((len(self.atoms), 3), dtype=complex)
            mode_av[self.indices] = u_av
            mode_Nav = np.vstack(N * [mode_av]) * phase_Na[:, np.newaxis]
            with Trajectory('%s.mode.%d.traj' % (self.name, l), 'w') as traj:
                for x in np.linspace(0, 2 * pi, nimages, endpoint=False):
                    atoms.set_positions((pos_Nav + np.exp(1j * x) * mode_Nav).real)
                    traj.write(atoms)