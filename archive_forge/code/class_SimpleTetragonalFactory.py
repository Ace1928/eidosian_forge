from ase.lattice.orthorhombic import (SimpleOrthorhombicFactory,
class SimpleTetragonalFactory(_Tetragonalize, SimpleOrthorhombicFactory):
    """A factory for creating simple tetragonal lattices."""
    orthobase = SimpleOrthorhombicFactory