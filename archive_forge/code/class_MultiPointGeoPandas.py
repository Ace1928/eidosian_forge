from __future__ import annotations
from packaging.version import Version
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.utils import isreal, ngjit
from numba import cuda
class MultiPointGeoPandas(_GeometryLike):

    @property
    def geom_dtypes(self):
        from geopandas.array import GeometryDtype
        return (GeometryDtype,)

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        import shapely
        geometry_name = self.geometry

        @ngjit
        @self.expand_aggs_and_cols(append)
        def _perform_extend_points(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols):
            x = values[j]
            y = values[j + 1]
            if xmin <= x <= xmax and ymin <= y <= ymax:
                xx = int(x_mapper(x) * sx + tx)
                yy = int(y_mapper(y) * sy + ty)
                xi, yi = (xx - 1 if x == xmax else xx, yy - 1 if y == ymax else yy)
                append(i, xi, yi, *aggs_and_cols)

        def extend(aggs, df, vt, bounds):
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            geometry = df[geometry_name].array
            ragged = shapely.to_ragged_array(geometry)
            geometry_type = ragged[0]
            if geometry_type not in (shapely.GeometryType.MULTIPOINT, shapely.GeometryType.POINT):
                raise ValueError(f'Canvas.points supports GeoPandas geometry types of POINT and MULTIPOINT, not {repr(geometry_type)}')
            coords = ragged[1].ravel()
            if geometry_type == shapely.GeometryType.POINT:
                extend_point_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, coords, *aggs_and_cols)
            else:
                offsets = ragged[2][0]
                extend_multipoint_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, coords, offsets, *aggs_and_cols)

        @ngjit
        @self.expand_aggs_and_cols(append)
        def extend_multipoint_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, offsets, *aggs_and_cols):
            for i in range(len(offsets) - 1):
                start = offsets[i]
                stop = offsets[i + 1]
                for j in range(start, stop):
                    _perform_extend_points(i, 2 * j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols)

        @ngjit
        @self.expand_aggs_and_cols(append)
        def extend_point_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols):
            n = len(values) // 2
            for i in range(n):
                _perform_extend_points(i, 2 * i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols)
        return extend