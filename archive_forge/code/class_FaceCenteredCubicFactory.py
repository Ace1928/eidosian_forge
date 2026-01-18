from ase.lattice.bravais import Bravais, reduceindex
import numpy as np
from ase.data import reference_states as _refstate
class FaceCenteredCubicFactory(SimpleCubicFactory):
    """A factory for creating face-centered cubic lattices."""
    xtal_name = 'fcc'
    int_basis = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    basis_factor = 0.5
    inverse_basis = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    inverse_basis_factor = 1.0
    atoms_in_unit_cell = 4