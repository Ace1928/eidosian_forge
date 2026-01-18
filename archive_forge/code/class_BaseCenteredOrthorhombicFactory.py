from ase.lattice.bravais import Bravais
import numpy as np
from ase.data import reference_states as _refstate
class BaseCenteredOrthorhombicFactory(SimpleOrthorhombicFactory):
    """A factory for creating base-centered orthorhombic lattices."""
    int_basis = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]])
    basis_factor = 0.5
    inverse_basis = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
    inverse_basis_factor = 1.0

    def check_basis_volume(self):
        """Check the volume of the unit cell."""
        vol1 = abs(np.linalg.det(self.basis))
        vol2 = self.calc_num_atoms() * np.linalg.det(self.latticeconstant) / 2.0
        if abs(vol1 - vol2) > 1e-05:
            print('WARNING: Got volume %f, expected %f' % (vol1, vol2))