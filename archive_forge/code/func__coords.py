import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry, JOIN_STYLE
from shapely.geometry.point import Point
def _coords(o):
    if isinstance(o, Point):
        return o.coords[0]
    else:
        return [float(c) for c in o]