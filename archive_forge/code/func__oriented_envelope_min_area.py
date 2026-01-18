import math
from itertools import islice
import numpy as np
import shapely
from shapely.affinity import affine_transform
def _oriented_envelope_min_area(geometry, **kwargs):
    """
    Computes the oriented envelope (minimum rotated rectangle) that encloses
    an input geometry.

    This is a fallback implementation for GEOS < 3.12 to have the correct
    minimum area behaviour.
    """
    if geometry is None:
        return None
    if geometry.is_empty:
        return shapely.from_wkt('POLYGON EMPTY')
    hull = geometry.convex_hull
    try:
        coords = hull.exterior.coords
    except AttributeError:
        return hull
    edges = ((pt2[0] - pt1[0], pt2[1] - pt1[1]) for pt1, pt2 in zip(coords, islice(coords, 1, None)))

    def _transformed_rects():
        for dx, dy in edges:
            length = math.sqrt(dx ** 2 + dy ** 2)
            ux, uy = (dx / length, dy / length)
            vx, vy = (-uy, ux)
            transf_rect = affine_transform(hull, (ux, uy, vx, vy, 0, 0)).envelope
            yield (transf_rect, (ux, vx, uy, vy, 0, 0))
    transf_rect, inv_matrix = min(_transformed_rects(), key=lambda r: r[0].area)
    return affine_transform(transf_rect, inv_matrix)