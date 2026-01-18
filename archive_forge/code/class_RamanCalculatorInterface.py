import numpy as np
import ase.units as un
class RamanCalculatorInterface(SiestaLRTDDFT):
    """Raman interface for Siesta calculator.
    When using the Raman calculator, please cite

    M. Walter and M. Moseler, Ab Initio Wavelength-Dependent Raman Spectra:
    Placzek Approximation and Beyond, J. Chem. Theory Comput. 2020, 16, 1, 576–586
    """

    def __init__(self, omega=0.0, **kw):
        """
        Parameters
        ----------
        omega: float
            frequency at which the Raman intensity should be computed, in eV

        kw: dictionary
            The parameter for the siesta_lrtddft object
        """
        self.omega = omega
        super().__init__(**kw)

    def __call__(self, *args, **kwargs):
        """Shorthand for calculate"""
        return self.calculate(*args, **kwargs)

    def calculate(self, atoms):
        """
        Calculate the polarizability for frequency omega

        Parameters
        ----------
        atoms: atoms class
            The atoms definition of the system. Not used but required by Raman
            calculator
        """
        pmat = self.get_polarizability(self.omega, Eext=np.array([1.0, 1.0, 1.0]))
        return pmat[:, :, 0].real * un.Bohr ** 2 / un.Ha