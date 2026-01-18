from abc import ABCMeta
from collections import OrderedDict
from functools import lru_cache
import io
import json
import math
import warnings
import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError
import shapely.geometry as sgeom
from shapely.prepared import prep
import cartopy.trace
def _project_polygon(self, polygon, src_crs):
    """
        Return the projected polygon(s) derived from the given polygon.

        """
    if src_crs.is_geodetic():
        is_ccw = True
    else:
        is_ccw = polygon.exterior.is_ccw
    rings = []
    multi_lines = []
    for src_ring in [polygon.exterior] + list(polygon.interiors):
        p_rings, p_mline = self._project_linear_ring(src_ring, src_crs)
        if p_rings:
            rings.extend(p_rings)
        if len(p_mline.geoms) > 0:
            multi_lines.append(p_mline)
    if multi_lines:
        rings.extend(self._attach_lines_to_boundary(multi_lines, is_ccw))
    return self._rings_to_multi_polygon(rings, is_ccw)