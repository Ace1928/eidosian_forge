from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
def _build_draw_trapezoid_y(append, map_onto_pixel, expand_aggs_and_cols):
    """Specialize a plotting kernel for drawing a trapezoid with two
    sides parallel to the y-axis"""

    @ngjit
    def clamp_y_indices(ystarti, ystopi, ymaxi):
        """Utility function to compute clamped y-indices"""
        out_of_bounds = ystarti < 0 and ystopi <= 0 or (ystarti > ymaxi and ystopi >= ymaxi)
        clamped_ystarti = max(0, min(ymaxi, ystarti))
        clamped_ystopi = max(-1, min(ymaxi + 1, ystopi))
        return (out_of_bounds, clamped_ystarti, clamped_ystopi)

    @ngjit
    @expand_aggs_and_cols
    def draw_trapezoid_y(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0, x1, y0, y1, y2, y3, trapezoid_start, stacked, *aggs_and_cols):
        """Draw a filled trapezoid that has two sides parallel to the y-axis

        Such a trapezoid is defined by two x coordinates (x0 for the left
        edge and x1 for the right parallel edge) and four y-coordinates
        (y0 for top left vertex, y1 for bottom left vertex, y2 for the bottom
        right vertex and y3 for the top right vertex).

                                          (x1, y3)
                                      _ +
                        (x0, y0)  _--   |
                                +       |
                                |       |
                                |       |
                                +       |
                        (x0, y1)  -     |
                                    -   |
                                      - |
                                        +
                                          (x1, y2)

        In a proper trapezoid (as drawn above), y0 >= y1 and y3 >= y2 so that
        edges do not intersect. This function also handles the case where
        y1 < y0 or y2 < y3, which results in a crossing edge.

        The trapezoid is filled using a vertical scan line algorithm where the
        start and stop bins are calculated by what amounts to a pair of
        Bresenham's line algorithms, one for the top edge and one for the
        bottom edge.

        Bins in the line connecting (x0, y1) and (x1, y2) are not filled if
        the `stacked` argument is set to True. This way stacked trapezoids
        will not have any overlapping bins.

        Parameters
        ----------
        x0, x1: float
            x-coordinate indices of the start and stop edge of the trapezoid
        y0, y1, y2, y3: float
            y-coordinate indices of the four corners of the trapezoid
        xmin, xmax, ymin, ymax: float
            The minimum and maximum allowable x and y value respectively.
            The trapezoid will be clipped to these values.
        i: int
            Group index
        trapezoid_start: bool
            If True, the filled trapezoid will include the (x0, y0) to (x0, y1)
            edge. Otherwise this edge will not be included.
        stacked: bool
            If False, the filled trapezoid will include the
            (x0, y1) to (x1, y2) edge. Otherwise this edge will not
            be included.
        """
        x0, x1, y0, y1, y2, y3, skip, clipped_start, clipped_end = _skip_or_clip_trapezoid_y(x0, x1, y0, y1, y2, y3, xmin, xmax, ymin, ymax)
        if skip:
            return
        x0i, y0i = map_onto_pixel(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0, y0)
        _, y1i = map_onto_pixel(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0, y1)
        x1i, y2i = map_onto_pixel(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x1, y2)
        _, y3i = map_onto_pixel(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x1, y3)
        xmaxi, ymaxi = map_onto_pixel(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xmax, ymax)
        dx = x1i - x0i
        ix = (dx > 0) - (dx < 0)
        dy0 = y3i - y0i
        iy0 = (dy0 > 0) - (dy0 < 0)
        dy1 = y2i - y1i
        iy1 = (dy1 > 0) - (dy1 < 0)
        trapezoid_start = trapezoid_start or clipped_start
        if trapezoid_start:
            y_oob, y_start, y_stop = clamp_y_indices(y0i, y1i, ymaxi)
            x_oob = x0i < 0 or x0i > xmaxi
            if y_oob or x_oob:
                pass
            elif y_start == y_stop and (not stacked):
                append(i, x0i, y_start, *aggs_and_cols)
            else:
                y = y_start
                iy = (y_start < y_stop) - (y_stop < y_start)
                if not stacked and -1 <= y_stop + iy <= ymaxi + 1:
                    y_stop += iy
                while y != y_stop:
                    append(i, x0i, y, *aggs_and_cols)
                    y += iy
        clipped = clipped_start or clipped_end
        if dx == 0 and (not clipped):
            y_oob, y_start, y_stop = clamp_y_indices(y3i, y2i, ymaxi)
            x_oob = x1i < 0 or x1i > xmaxi
            if y_oob or x_oob:
                pass
            elif y_start == y_stop and (not stacked):
                append(i, x1i, y_start, *aggs_and_cols)
            else:
                y = y_start
                iy = (y_start < y_stop) - (y_stop < y_start)
                if not stacked and -1 <= y_stop + iy <= ymaxi + 1:
                    y_stop += iy
                while y != y_stop:
                    append(i, x1i, y, *aggs_and_cols)
                    y += iy
            return
        dx = abs(dx) * 2
        dy0 = abs(dy0) * 2
        dy1 = abs(dy1) * 2
        error0 = 2 * dy0 - dx
        error1 = 2 * dy1 - dx
        while x0i != x1i:
            while error0 >= 0 and (error0 or ix > 0):
                error0 -= 2 * dx
                y0i += iy0
            error0 += 2 * dy0
            while error1 >= 0 and (error1 or ix > 0):
                error1 -= 2 * dx
                y1i += iy1
            error1 += 2 * dy1
            x0i += ix
            x_oob = x0i < 0 or x0i > xmaxi
            if x_oob:
                continue
            y_oob, y_start, y_stop = clamp_y_indices(y0i, y1i, ymaxi)
            if y_oob:
                pass
            elif y_start == y_stop and (not stacked):
                append(i, x0i, y_start, *aggs_and_cols)
            else:
                y = y_start
                iy = (y_start < y_stop) - (y_stop < y_start)
                if not stacked and -1 <= y_stop + iy <= ymaxi + 1:
                    y_stop += iy
                while y != y_stop:
                    append(i, x0i, y, *aggs_and_cols)
                    y += iy
    return draw_trapezoid_y