import os
import sys
import numpy as np
from ase import units
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