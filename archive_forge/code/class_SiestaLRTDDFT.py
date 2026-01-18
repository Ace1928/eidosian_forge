import numpy as np
import ase.units as un
class SiestaLRTDDFT:
    """Interface for linear response TDDFT for Siesta via `PyNAO`_

    When using PyNAO please cite the papers indicated in the
    `documentation <https://mbarbrywebsite.ddns.net/pynao/doc/html/references.html>`_
    """

    def __init__(self, initialize=False, **kw):
        """
        Parameters
        ----------
        initialize: bool
            To initialize the tddft calculations before calculating the polarizability
            Can be useful to calculate multiple frequency range without the need
            to recalculate the kernel
        kw: dictionary
            keywords for the tddft_iter function from PyNAO
        """
        try:
            from pynao import tddft_iter
        except ModuleNotFoundError as err:
            msg = 'running lrtddft with Siesta calculator requires pynao package'
            raise ModuleNotFoundError(msg) from err
        self.initialize = initialize
        self.lrtddft_params = kw
        self.tddft = None
        if 'iter_broadening' in self.lrtddft_params:
            self.lrtddft_params['iter_broadening'] /= un.Ha
        if self.initialize:
            self.tddft = tddft_iter(**self.lrtddft_params)

    def get_ground_state(self, atoms, **kw):
        """
        Run siesta calculations in order to get ground state properties.
        Makes sure that the proper parameters are unsed in order to be able
        to run pynao afterward, i.e.,

            COOP.Write = True
            WriteDenchar = True
            XML.Write = True
        """
        from ase.calculators.siesta import Siesta
        if 'fdf_arguments' not in kw.keys():
            kw['fdf_arguments'] = {'COOP.Write': True, 'WriteDenchar': True, 'XML.Write': True}
        else:
            for param in ['COOP.Write', 'WriteDenchar', 'XML.Write']:
                kw['fdf_arguments'][param] = True
        siesta = Siesta(**kw)
        atoms.calc = siesta
        atoms.get_potential_energy()

    def get_polarizability(self, omega, Eext=np.array([1.0, 1.0, 1.0]), inter=True):
        """
        Calculate the polarizability of a molecule via linear response TDDFT
        calculation.

        Parameters
        ----------
        omega: float or array like
            frequency range for which the polarizability should be computed, in eV

        Returns
        -------
        polarizability: array like (complex)
            array of dimension (3, 3, nff) with nff the number of frequency,
            the first and second dimension are the matrix elements of the
            polarizability in atomic units::

                P_xx, P_xy, P_xz, Pyx, .......

        Example
        -------

        from ase.calculators.siesta.siesta_lrtddft import siestaLRTDDFT
        from ase.build import molecule
        import numpy as np
        import matplotlib.pyplot as plt

        # Define the systems
        CH4 = molecule('CH4')

        lr = siestaLRTDDFT(label="siesta", jcutoff=7, iter_broadening=0.15,
                            xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7)

        # run DFT calculation with Siesta
        lr.get_ground_state(CH4)

        # run TDDFT calculation with PyNAO
        freq=np.arange(0.0, 25.0, 0.05)
        pmat = lr.get_polarizability(freq) 
        """
        from pynao import tddft_iter
        if not self.initialize:
            self.tddft = tddft_iter(**self.lrtddft_params)
        if isinstance(omega, float):
            freq = np.array([omega])
        elif isinstance(omega, list):
            freq = np.array([omega])
        elif isinstance(omega, np.ndarray):
            freq = omega
        else:
            raise ValueError('omega soulf')
        freq_cmplx = freq / un.Ha + 1j * self.tddft.eps
        if inter:
            pmat = -self.tddft.comp_polariz_inter_Edir(freq_cmplx, Eext=Eext)
            self.dn = self.tddft.dn
        else:
            pmat = -self.tddft.comp_polariz_nonin_Edir(freq_cmplx, Eext=Eext)
            self.dn = self.tddft.dn0
        return pmat