from ase.lattice.bravais import Bravais
import numpy as np
from ase.data import reference_states as _refstate
class FaceCenteredOrthorhombicFactory(SimpleOrthorhombicFactory):
    """A factory for creating face-centered orthorhombic lattices."""
    int_basis = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    basis_factor = 0.5
    inverse_basis = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    inverse_basis_factor = 1.0

    def check_basis_volume(self):
        """Check the volume of the unit cell."""
        vol1 = abs(np.linalg.det(self.basis))
        vol2 = self.calc_num_atoms() * np.linalg.det(self.latticeconstant) / 4.0
        if abs(vol1 - vol2) > 1e-05:
            print('WARNING: Got volume %f, expected %f' % (vol1, vol2))