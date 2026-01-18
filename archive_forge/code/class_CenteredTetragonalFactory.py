from ase.lattice.orthorhombic import (SimpleOrthorhombicFactory,
class CenteredTetragonalFactory(_Tetragonalize, BodyCenteredOrthorhombicFactory):
    """A factory for creating centered tetragonal lattices."""
    orthobase = BodyCenteredOrthorhombicFactory