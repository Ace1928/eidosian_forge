from ase.lattice.triclinic import TriclinicFactory
class GrapheneFactory(HexagonalFactory):
    """A factory for creating graphene lattices."""
    xtal_name = 'graphene'
    bravais_basis = [[0, 0, 0], [1.0 / 3.0, 2.0 / 3.0, 0]]