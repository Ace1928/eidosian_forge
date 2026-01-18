from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
def _get_all_x(self, distance):
    """Helper function to get bonds, angles, dihedrals"""
    maxIter = self.nImages
    if len(self.nl) == 1:
        maxIter = 1
    xList = []
    for i in range(maxIter):
        xList.append(get_distance_indices(self.distance_matrix[i], distance))
    return xList