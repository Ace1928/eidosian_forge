from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
def _line_internal_build_extend(x_mapper, y_mapper, append, line_width, antialias_stage_2, antialias_stage_2_funcs, expand_aggs_and_cols):
    antialias = line_width > 0
    map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper, antialias)
    overwrite, use_2_stage_agg = two_stage_agg(antialias_stage_2)
    if not use_2_stage_agg:
        antialias_stage_2_funcs = None
    draw_segment = _build_draw_segment(append, map_onto_pixel, expand_aggs_and_cols, line_width, overwrite)
    return (draw_segment, antialias_stage_2_funcs)