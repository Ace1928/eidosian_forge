import numpy as np
import shapely
from shapely.algorithms.cga import is_ccw_impl, signed_area
from shapely.errors import TopologicalError
from shapely.geometry.base import BaseGeometry
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
def _unpickle_linearring(wkb):
    linestring = shapely.from_wkb(wkb)
    srid = shapely.get_srid(linestring)
    linearring = shapely.linearrings(shapely.get_coordinates(linestring))
    if srid:
        linearring = shapely.set_srid(linearring, srid)
    return linearring