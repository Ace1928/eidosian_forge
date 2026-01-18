from .cusps import CuspPostDrillInfo
from .geometric_structure import compute_r13_planes_for_tet
from .tracing import compute_plane_intersection_param, Endpoint, GeodesicPiece
from .epsilons import compute_epsilon
from . import constants
from . import exceptions
from ..snap.t3mlite import simplex, Perm4, Tetrahedron # type: ignore
from ..matrix import matrix # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence, Optional, Union, Tuple, List, Dict, Mapping
def _retrace_geodesic_piece(index: int, tets: Union[Mapping[int, Tetrahedron], Sequence[Tetrahedron]], tet: Tetrahedron, face: int, dimension_end_cell: int, points, trace_direction: int, verified: bool, allowed_end_corners: Optional[Sequence[Tuple[Tetrahedron, int]]]=None):
    start_point, end_point = points
    RF = start_point[0].parent()
    if verified:
        epsilon = 0
    else:
        epsilon = compute_epsilon(RF)
    direction = end_point - start_point
    pieces: List[GeodesicPiece] = []
    param = RF(0)
    for i in range(4):
        hit_face: Optional[int] = None
        hit_param = None
        for candidate_face, plane in tet.R13_unnormalised_planes.items():
            if candidate_face == face:
                continue
            candidate_param = compute_plane_intersection_param(plane, start_point, direction, verified)
            if candidate_param < param - epsilon:
                continue
            if not candidate_param > param + epsilon:
                raise InsufficientPrecisionError('When re-tracing the geodesic, the intersection with the next tetrahedron face was too close to the previous to tell them apart. Increasing the precision will probably avoid this problem.')
            if hit_param is None:
                hit_param = candidate_param
                hit_face = candidate_face
            elif candidate_param + epsilon < hit_param:
                hit_param = candidate_param
                hit_face = candidate_face
            elif not candidate_param > hit_param + epsilon:
                raise exceptions.RetracingRayHittingOneSkeletonError()
        if hit_param is None or hit_face is None:
            raise InsufficientPrecisionError('Could not find the next intersection of the geodesic with a tetrahedron face. Increasing the precision should solve this problem.')
        if dimension_end_cell == 3:
            if hit_param < RF(1) - epsilon:
                hit_end: bool = False
            elif hit_param > RF(1) + epsilon:
                hit_end = True
            else:
                raise InsufficientPrecisionError('Could not determine whether we finished re-tracing geodesic piece.Increasing the precision will most likely fix the problem.')
        elif len(tets) == 3:
            hit_end = hit_face in [simplex.F0, simplex.F1]
        else:
            hit_end = tet is tets[hit_face]
        if hit_end:
            if dimension_end_cell == 3:
                end_cell: int = simplex.T
            else:
                end_cell = hit_face
            if allowed_end_corners:
                if not (tet, end_cell) in allowed_end_corners:
                    raise Exception('Re-tracing geodesic piece ended at wrong cell. This is either due to a lack of precision or an implementation bug.')
            endpoints = [Endpoint(start_point + param * direction, face), Endpoint(end_point, end_cell)][::trace_direction]
            pieces.append(GeodesicPiece.create_and_attach(index, tet, endpoints))
            break
        pieces.append(GeodesicPiece.create_and_attach(index, tet, [Endpoint(start_point + param * direction, face), Endpoint(start_point + hit_param * direction, hit_face)][::trace_direction]))
        face = tet.Gluing[hit_face].image(hit_face)
        tet = tet.Neighbor[hit_face]
        param = hit_param
        if dimension_end_cell == 0:
            if allowed_end_corners:
                if not (tet, simplex.comp(face)) in allowed_end_corners:
                    raise Exception('Re-tracing geodesic piece ended at wrong cell. This is either due to a lack of precision or an implementation bug.')
            pieces.append(GeodesicPiece.create_face_to_vertex_and_attach(index, tet, Endpoint(start_point + hit_param * direction, face), trace_direction))
            break
    else:
        raise Exception('Too many steps when re-tracing a geodesic piece. This is either due to a lack of precision or an implementation bug.')
    return pieces[::trace_direction]