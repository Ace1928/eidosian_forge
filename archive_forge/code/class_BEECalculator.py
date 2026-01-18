import numpy as np
import pytest
from ase.dft.bee import BEEFEnsemble, ensemble, readbee
class BEECalculator:
    """Fake DFT calculator."""
    atoms = None

    def __init__(self, name):
        self.name = name

    def get_xc_functional(self):
        return self.name

    def get_nonselfconsistent_energies(self, beef_type: str) -> np.ndarray:
        n = {'mbeef': 64, 'beefvdw': 32, 'mbeefvdw': 28}[beef_type]
        return np.linspace(-1, 1, n)

    def get_potential_energy(self, atoms):
        return 0.0