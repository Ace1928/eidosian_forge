from ase.lattice.triclinic import TriclinicFactory
class HexagonalClosedPackedFactory(HexagonalFactory):
    """A factory for creating HCP lattices."""
    xtal_name = 'hcp'
    bravais_basis = [[0, 0, 0], [1.0 / 3.0, 2.0 / 3.0, 0.5]]