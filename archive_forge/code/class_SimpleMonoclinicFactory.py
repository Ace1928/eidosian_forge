from ase.lattice.triclinic import TriclinicFactory
import numpy as np
class SimpleMonoclinicFactory(TriclinicFactory):
    """A factory for creating simple monoclinic lattices."""
    xtal_name = 'monoclinic'

    def make_crystal_basis(self):
        """Make the basis matrix for the crystal unit cell and the system unit cell."""
        if isinstance(self.latticeconstant, type({})):
            self.latticeconstant['beta'] = 90
            self.latticeconstant['gamma'] = 90
        elif len(self.latticeconstant) == 4:
            self.latticeconstant = self.latticeconstant + (90, 90)
        else:
            raise ValueError('Improper lattice constants for monoclinic crystal.')
        TriclinicFactory.make_crystal_basis(self)