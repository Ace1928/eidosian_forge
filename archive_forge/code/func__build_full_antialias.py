from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
def _build_full_antialias(expand_aggs_and_cols):
    """Specialize antialiased line drawing algorithm for a given append/axis combination"""

    @ngjit
    @expand_aggs_and_cols
    def _full_antialias(line_width, overwrite, i, x0, x1, y0, y1, segment_start, segment_end, xm, ym, append, nx, ny, buffer, *aggs_and_cols):
        """Draw an antialiased line segment.

        If overwrite=True can overwrite each pixel multiple times because
        using max for the overwriting.  If False can only write each pixel
        once per segment and its previous segment.
        Argument xm, ym are only valid if overwrite and segment_start are False.
        """
        if x0 == x1 and y0 == y1:
            return
        flip_xy = abs(x0 - x1) < abs(y0 - y1)
        if flip_xy:
            x0, y0 = (y0, x0)
            x1, y1 = (y1, x1)
            xm, ym = (ym, xm)
        scale = 1.0
        if line_width < 1.0:
            scale *= line_width
            line_width = 1.0
        aa = 1.0
        halfwidth = 0.5 * (line_width + aa)
        flip_order = y1 < y0 or (y1 == y0 and x1 < x0)
        alongx = float(x1 - x0)
        alongy = float(y1 - y0)
        length = math.sqrt(alongx ** 2 + alongy ** 2)
        alongx /= length
        alongy /= length
        rightx = alongy
        righty = -alongx
        if flip_order:
            buffer[0] = x1 - halfwidth * (rightx - alongx)
            buffer[1] = x1 - halfwidth * (-rightx - alongx)
            buffer[2] = x0 - halfwidth * (-rightx + alongx)
            buffer[3] = x0 - halfwidth * (rightx + alongx)
            buffer[4] = y1 - halfwidth * (righty - alongy)
            buffer[5] = y1 - halfwidth * (-righty - alongy)
            buffer[6] = y0 - halfwidth * (-righty + alongy)
            buffer[7] = y0 - halfwidth * (righty + alongy)
        else:
            buffer[0] = x0 + halfwidth * (rightx - alongx)
            buffer[1] = x0 + halfwidth * (-rightx - alongx)
            buffer[2] = x1 + halfwidth * (-rightx + alongx)
            buffer[3] = x1 + halfwidth * (rightx + alongx)
            buffer[4] = y0 + halfwidth * (righty - alongy)
            buffer[5] = y0 + halfwidth * (-righty - alongy)
            buffer[6] = y1 + halfwidth * (-righty + alongy)
            buffer[7] = y1 + halfwidth * (righty + alongy)
        xmax = nx - 1
        ymax = ny - 1
        if flip_xy:
            xmax, ymax = (ymax, xmax)
        if flip_order:
            lowindex = 0 if x0 > x1 else 1
        else:
            lowindex = 0 if x1 > x0 else 1
        if not overwrite and (not segment_start):
            prev_alongx = x0 - xm
            prev_alongy = y0 - ym
            prev_length = math.sqrt(prev_alongx ** 2 + prev_alongy ** 2)
            if prev_length > 0.0:
                prev_alongx /= prev_length
                prev_alongy /= prev_length
                prev_rightx = prev_alongy
                prev_righty = -prev_alongx
            else:
                overwrite = True
        ystart = _clamp(math.ceil(buffer[4 + lowindex]), 0, ymax)
        yend = _clamp(math.floor(buffer[4 + (lowindex + 2) % 4]), 0, ymax)
        ll = lowindex
        lu = (ll + 1) % 4
        rl = lowindex
        ru = (rl + 3) % 4
        for y in range(ystart, yend + 1):
            if ll == lowindex and y > buffer[4 + lu]:
                ll = lu
                lu = (ll + 1) % 4
            if rl == lowindex and y > buffer[4 + ru]:
                rl = ru
                ru = (rl + 3) % 4
            xleft = _clamp(math.ceil(_x_intercept(y, buffer[ll], buffer[4 + ll], buffer[lu], buffer[4 + lu])), 0, xmax)
            xright = _clamp(math.floor(_x_intercept(y, buffer[rl], buffer[4 + rl], buffer[ru], buffer[4 + ru])), 0, xmax)
            for x in range(xleft, xright + 1):
                along = (x - x0) * alongx + (y - y0) * alongy
                prev_correction = False
                if along < 0.0:
                    if overwrite or segment_start or (x - x0) * prev_alongx + (y - y0) * prev_alongy > 0.0:
                        distance = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                    else:
                        continue
                elif along > length:
                    if overwrite or segment_end:
                        distance = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
                    else:
                        continue
                else:
                    distance = abs((x - x0) * rightx + (y - y0) * righty)
                    if not overwrite and (not segment_start) and (-prev_length <= (x - x0) * prev_alongx + (y - y0) * prev_alongy <= 0.0) and (abs((x - x0) * prev_rightx + (y - y0) * prev_righty) <= halfwidth):
                        prev_correction = True
                value = 1.0 - _linearstep(0.5 * (line_width - aa), halfwidth, distance)
                value *= scale
                prev_value = 0.0
                if prev_correction:
                    prev_distance = abs((x - x0) * prev_rightx + (y - y0) * prev_righty)
                    prev_value = 1.0 - _linearstep(0.5 * (line_width - aa), halfwidth, prev_distance)
                    prev_value *= scale
                    if value <= prev_value:
                        value = 0.0
                if value > 0.0:
                    xx, yy = (y, x) if flip_xy else (x, y)
                    append(i, xx, yy, value, prev_value, *aggs_and_cols)
    return _full_antialias