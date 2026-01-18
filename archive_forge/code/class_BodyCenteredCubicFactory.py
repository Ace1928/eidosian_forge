from ase.lattice.bravais import Bravais, reduceindex
import numpy as np
from ase.data import reference_states as _refstate
class BodyCenteredCubicFactory(SimpleCubicFactory):
    """A factory for creating body-centered cubic lattices."""
    xtal_name = 'bcc'
    int_basis = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    basis_factor = 0.5
    inverse_basis = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    inverse_basis_factor = 1.0
    atoms_in_unit_cell = 2