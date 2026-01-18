from ..snap.t3mlite import simplex
from ..hyperboloid import *
def _find_all_tetrahedra(tet):
    result = []
    pending_tets = [tet]
    visited_tets = set()
    while pending_tets:
        tet = pending_tets.pop()
        if tet not in visited_tets:
            visited_tets.add(tet)
            result.append(tet)
            for neighbor in tet.Neighbor.values():
                pending_tets.append(neighbor)
    return result