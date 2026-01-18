from toolz import memoize
import numpy as np
from datashader.glyphs.line import _build_map_onto_pixel_for_line
from datashader.glyphs.points import _GeometryLike
from datashader.utils import ngjit
def _build_extend_geopandas_polygon_geometry(draw_polygon, expand_aggs_and_cols):
    import shapely

    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, geometry, *aggs_and_cols):
        ragged = shapely.to_ragged_array(geometry)
        geometry_type = ragged[0]
        if geometry_type not in (shapely.GeometryType.POLYGON, shapely.GeometryType.MULTIPOLYGON):
            raise ValueError(f'Canvas.polygons supports GeoPandas geometry types of POLYGON and MULTIPOLYGON, not {repr(geometry_type)}')
        coords = ragged[1].ravel()
        if geometry_type == shapely.GeometryType.MULTIPOLYGON:
            offsets0, offsets1, offsets2 = ragged[2]
        else:
            offsets0, offsets1 = ragged[2]
            offsets2 = np.arange(len(offsets1))
        extend_cpu_numba(sx, tx, sy, ty, xmin, xmax, ymin, ymax, coords, offsets0, offsets1, offsets2, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba(sx, tx, sy, ty, xmin, xmax, ymin, ymax, coords, offsets0, offsets1, offsets2, *aggs_and_cols):
        max_edges = 0
        n_multipolygons = len(offsets2) - 1
        for i in range(n_multipolygons):
            polygon_inds = offsets1[offsets2[i]:offsets2[i + 1] + 1]
            for j in range(len(polygon_inds) - 1):
                start = offsets0[polygon_inds[j]]
                stop = offsets0[polygon_inds[j + 1]]
                max_edges = max(max_edges, stop - start - 1)
        xs = np.full((max_edges, 2), np.nan, dtype=np.float32)
        ys = np.full((max_edges, 2), np.nan, dtype=np.float32)
        yincreasing = np.zeros(max_edges, dtype=np.int8)
        eligible = np.ones(max_edges, dtype=np.int8)
        for i in range(n_multipolygons):
            polygon_inds = offsets1[offsets2[i]:offsets2[i + 1] + 1]
            for j in range(len(polygon_inds) - 1):
                start = polygon_inds[j]
                stop = polygon_inds[j + 1]
                draw_polygon(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, offsets0[start:stop + 1], 2, coords, xs, ys, yincreasing, eligible, *aggs_and_cols)
    return extend_cpu