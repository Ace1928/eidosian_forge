from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
def _idxTuple2SymbolTuple(self, imI, tup):
    """Converts a tuple of indices to their symbols"""
    return (self.images[imI][idx].symbol for idx in tup)