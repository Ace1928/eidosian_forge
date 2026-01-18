from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
def _build_extend_line_axis1_geopandas(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs):
    if antialias_stage_2_funcs is not None:
        aa_stage_2_accumulate, aa_stage_2_clear, aa_stage_2_copy_back = antialias_stage_2_funcs
    use_2_stage_agg = antialias_stage_2_funcs is not None
    import shapely

    def _process_geometry(geometry):
        ragged = shapely.to_ragged_array(geometry)
        geometry_type = ragged[0]
        if geometry_type not in (shapely.GeometryType.LINESTRING, shapely.GeometryType.MULTILINESTRING, shapely.GeometryType.MULTIPOLYGON, shapely.GeometryType.POLYGON):
            raise ValueError(f'Canvas.line supports GeoPandas geometry types of LINESTRING, MULTILINESTRING, MULTIPOLYGON and POLYGON, not {repr(geometry_type)}')
        coords = ragged[1].ravel()
        if geometry_type == shapely.GeometryType.LINESTRING:
            offsets = ragged[2][0]
            outer_offsets = np.arange(len(offsets))
            closed_rings = False
        elif geometry_type == shapely.GeometryType.MULTILINESTRING:
            offsets, outer_offsets = ragged[2]
            closed_rings = False
        elif geometry_type == shapely.GeometryType.MULTIPOLYGON:
            offsets, temp_offsets, outer_offsets = ragged[2]
            outer_offsets = temp_offsets[outer_offsets]
            closed_rings = True
        else:
            offsets, outer_offsets = ragged[2]
            closed_rings = True
        return (coords, offsets, outer_offsets, closed_rings)

    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, geometry, antialias_stage_2, *aggs_and_cols):
        coords, offsets, outer_offsets, closed_rings = _process_geometry(geometry)
        extend_cpu_numba(sx, tx, sy, ty, xmin, xmax, ymin, ymax, coords, offsets, outer_offsets, closed_rings, antialias_stage_2, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, offsets, outer_offsets, closed_rings, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        n_multilines = len(outer_offsets) - 1
        for i in range(n_multilines):
            start0 = outer_offsets[i]
            stop0 = outer_offsets[i + 1]
            for j in range(start0, stop0):
                start1 = offsets[j]
                stop1 = offsets[j + 1]
                for k in range(2 * start1, 2 * stop1 - 2, 2):
                    x0 = values[k]
                    y0 = values[k + 1]
                    x1 = values[k + 2]
                    y1 = values[k + 3]
                    if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
                        continue
                    segment_start = k == start1 and (not closed_rings) or (k > start1 and (not np.isfinite(values[k - 2]) or not np.isfinite(values[k - 1])))
                    segment_end = not closed_rings and k == stop1 - 4 or (k < stop1 - 4 and (not np.isfinite(values[k + 4]) or not np.isfinite(values[k + 5])))
                    if segment_start or use_2_stage_agg:
                        xm = 0.0
                        ym = 0.0
                    elif k == start1 and closed_rings:
                        xm = values[stop1 - 4]
                        ym = values[stop1 - 3]
                    else:
                        xm = values[k - 2]
                        ym = values[k - 1]
                    draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, segment_end, x0, x1, y0, y1, xm, ym, buffer, *aggs_and_cols)

    def extend_cpu_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, geometry, antialias_stage_2, *aggs_and_cols):
        coords, offsets, outer_offsets, closed_rings = _process_geometry(geometry)
        n_aggs = len(antialias_stage_2[0])
        aggs_and_accums = tuple(((agg, agg.copy()) for agg in aggs_and_cols[:n_aggs]))
        extend_cpu_numba_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, coords, offsets, outer_offsets, closed_rings, antialias_stage_2, aggs_and_accums, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, offsets, outer_offsets, closed_rings, antialias_stage_2, aggs_and_accums, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        n_multilines = len(outer_offsets) - 1
        first_pass = True
        for i in range(n_multilines):
            start0 = outer_offsets[i]
            stop0 = outer_offsets[i + 1]
            for j in range(start0, stop0):
                start1 = offsets[j]
                stop1 = offsets[j + 1]
                for k in range(2 * start1, 2 * stop1 - 2, 2):
                    x0 = values[k]
                    y0 = values[k + 1]
                    x1 = values[k + 2]
                    y1 = values[k + 3]
                    if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
                        continue
                    segment_start = k == start1 and (not closed_rings) or (k > start1 and (not np.isfinite(values[k - 2]) or not np.isfinite(values[k - 1])))
                    segment_end = not closed_rings and k == stop1 - 4 or (k < stop1 - 4 and (not np.isfinite(values[k + 4]) or not np.isfinite(values[k + 5])))
                    draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, segment_end, x0, x1, y0, y1, 0.0, 0.0, buffer, *aggs_and_cols)
            aa_stage_2_accumulate(aggs_and_accums, first_pass)
            first_pass = False
            aa_stage_2_clear(aggs_and_accums)
        aa_stage_2_copy_back(aggs_and_accums)
    if use_2_stage_agg:
        return extend_cpu_antialias_2agg
    else:
        return extend_cpu