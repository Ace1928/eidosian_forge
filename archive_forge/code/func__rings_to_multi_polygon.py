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
def _rings_to_multi_polygon(self, rings, is_ccw):
    exterior_rings = []
    interior_rings = []
    for ring in rings:
        if ring.is_ccw != is_ccw:
            interior_rings.append(ring)
        else:
            exterior_rings.append(ring)
    polygon_bits = []
    for exterior_ring in exterior_rings:
        polygon = sgeom.Polygon(exterior_ring)
        prep_polygon = prep(polygon)
        holes = []
        for interior_ring in interior_rings[:]:
            if prep_polygon.contains(interior_ring):
                holes.append(interior_ring)
                interior_rings.remove(interior_ring)
            elif polygon.crosses(interior_ring):
                holes.append(interior_ring)
                interior_rings.remove(interior_ring)
        polygon_bits.append((exterior_ring.coords, [ring.coords for ring in holes]))
    if interior_rings:
        boundary_poly = self.domain
        x3, y3, x4, y4 = boundary_poly.bounds
        bx = (x4 - x3) * 0.1
        by = (y4 - y3) * 0.1
        x3 -= bx
        y3 -= by
        x4 += bx
        y4 += by
        for ring in interior_rings:
            polygon = sgeom.Polygon(ring).buffer(0)
            if not polygon.is_empty and polygon.is_valid:
                x1, y1, x2, y2 = polygon.bounds
                bx = (x2 - x1) * 0.1
                by = (y2 - y1) * 0.1
                x1 -= bx
                y1 -= by
                x2 += bx
                y2 += by
                box = sgeom.box(min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4))
                polygon = box.difference(polygon)
                polygon = boundary_poly.intersection(polygon)
                if not polygon.is_empty:
                    polygon_bits.append(polygon)
    if polygon_bits:
        multi_poly = sgeom.MultiPolygon(polygon_bits)
    else:
        multi_poly = sgeom.MultiPolygon()
    return multi_poly