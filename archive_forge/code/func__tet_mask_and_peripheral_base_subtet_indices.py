from .cusps import CuspPostDrillInfo
from .tracing import GeodesicPiece
from .peripheral_curves import install_peripheral_curves
from ..snap.t3mlite import Tetrahedron, Perm4, Mcomplex, simplex
from typing import Dict, Tuple, List, Sequence
def _tet_mask_and_peripheral_base_subtet_indices(tetrahedra):
    """
    Given the same input as described in crush_geodesic_data,
    computes the bit mask of which subtetrahedra will not be
    crushed. Also return a set of indices, that is the index
    of one subtetrahedron for each simple closed curve that
    can be used later to compute a new meridian and longitude.
    """
    mask = 24 * len(tetrahedra) * [True]
    index_to_peripheral_base_subtet_index = {}
    for tet in tetrahedra:
        for piece in tet.geodesic_pieces:
            perm = _find_perm_for_piece(piece)
            _traverse_edge(tet, perm, mask)
            if piece.index not in index_to_peripheral_base_subtet_index:
                other_perm = perm * _transpositions[1]
                subtet_index = 24 * tet.Index + _perm_to_index(other_perm)
                index_to_peripheral_base_subtet_index[piece.index] = subtet_index
    return (mask, index_to_peripheral_base_subtet_index.values())