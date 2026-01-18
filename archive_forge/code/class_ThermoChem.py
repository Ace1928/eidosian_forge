import os
import sys
import numpy as np
from ase import units
class ThermoChem:
    """Base class containing common methods used in thermochemistry
    calculations."""

    def get_ZPE_correction(self):
        """Returns the zero-point vibrational energy correction in eV."""
        zpe = 0.0
        for energy in self.vib_energies:
            zpe += 0.5 * energy
        return zpe

    def _vibrational_energy_contribution(self, temperature):
        """Calculates the change in internal energy due to vibrations from
        0K to the specified temperature for a set of vibrations given in
        eV and a temperature given in Kelvin. Returns the energy change
        in eV."""
        kT = units.kB * temperature
        dU = 0.0
        for energy in self.vib_energies:
            dU += energy / (np.exp(energy / kT) - 1.0)
        return dU

    def _vibrational_entropy_contribution(self, temperature):
        """Calculates the entropy due to vibrations for a set of vibrations
        given in eV and a temperature given in Kelvin.  Returns the entropy
        in eV/K."""
        kT = units.kB * temperature
        S_v = 0.0
        for energy in self.vib_energies:
            x = energy / kT
            S_v += x / (np.exp(x) - 1.0) - np.log(1.0 - np.exp(-x))
        S_v *= units.kB
        return S_v

    def _vprint(self, text):
        """Print output if verbose flag True."""
        if self.verbose:
            sys.stdout.write(text + os.linesep)