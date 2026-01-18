import os
import sys
import numpy as np
from ase import units
class HarmonicThermo(ThermoChem):
    """Class for calculating thermodynamic properties in the approximation
    that all degrees of freedom are treated harmonically. Often used for
    adsorbates.

    Inputs:

    vib_energies : list
        a list of the harmonic energies of the adsorbate (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of
        energies should match the number of degrees of freedom of the
        adsorbate; i.e., 3*n, where n is the number of atoms. Note that
        this class does not check that the user has supplied the correct
        number of energies. Units of energies are eV.
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)
    """

    def __init__(self, vib_energies, potentialenergy=0.0):
        self.vib_energies = vib_energies
        if sum(np.iscomplex(self.vib_energies)):
            raise ValueError('Imaginary vibrational energies are present.')
        else:
            self.vib_energies = np.real(self.vib_energies)
        self.potentialenergy = potentialenergy

    def get_internal_energy(self, temperature, verbose=True):
        """Returns the internal energy, in eV, in the harmonic approximation
        at a specified temperature (K)."""
        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.3f eV'
        write('Internal energy components at T = %.2f K:' % temperature)
        write('=' * 31)
        U = 0.0
        write(fmt % ('E_pot', self.potentialenergy))
        U += self.potentialenergy
        zpe = self.get_ZPE_correction()
        write(fmt % ('E_ZPE', zpe))
        U += zpe
        dU_v = self._vibrational_energy_contribution(temperature)
        write(fmt % ('Cv_harm (0->T)', dU_v))
        U += dU_v
        write('-' * 31)
        write(fmt % ('U', U))
        write('=' * 31)
        return U

    def get_entropy(self, temperature, verbose=True):
        """Returns the entropy, in eV/K, in the harmonic approximation
        at a specified temperature (K)."""
        self.verbose = verbose
        write = self._vprint
        fmt = '%-15s%13.7f eV/K%13.3f eV'
        write('Entropy components at T = %.2f K:' % temperature)
        write('=' * 49)
        write('%15s%13s     %13s' % ('', 'S', 'T*S'))
        S = 0.0
        S_v = self._vibrational_entropy_contribution(temperature)
        write(fmt % ('S_harm', S_v, S_v * temperature))
        S += S_v
        write('-' * 49)
        write(fmt % ('S', S, S * temperature))
        write('=' * 49)
        return S

    def get_helmholtz_energy(self, temperature, verbose=True):
        """Returns the Helmholtz free energy, in eV, in the harmonic
        approximation at a specified temperature (K)."""
        self.verbose = True
        write = self._vprint
        U = self.get_internal_energy(temperature, verbose=verbose)
        write('')
        S = self.get_entropy(temperature, verbose=verbose)
        F = U - temperature * S
        write('')
        write('Free energy components at T = %.2f K:' % temperature)
        write('=' * 23)
        fmt = '%5s%15.3f eV'
        write(fmt % ('U', U))
        write(fmt % ('-T*S', -temperature * S))
        write('-' * 23)
        write(fmt % ('F', F))
        write('=' * 23)
        return F