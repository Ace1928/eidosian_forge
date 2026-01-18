from ase.lattice.orthorhombic import (SimpleOrthorhombicFactory,
class _Tetragonalize:
    """A mixin class for implementing tetragonal crystals as orthorhombic ones."""
    xtal_name = 'tetragonal'

    def make_crystal_basis(self):
        lattice = self.latticeconstant
        if isinstance(lattice, type({})):
            lattice['b/a'] = 1.0
        elif len(lattice) == 2:
            lattice = (lattice[0], lattice[0], lattice[1])
        else:
            raise ValueError('Improper lattice constants for tetragonal crystal.')
        self.latticeconstant = lattice
        self.orthobase.make_crystal_basis(self)