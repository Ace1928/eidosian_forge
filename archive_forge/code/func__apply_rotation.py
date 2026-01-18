from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def _apply_rotation(point, degrees, centroid):
    return (round(point[0] * math.cos(math.radians(360 - degrees)) - point[1] * math.sin(math.radians(360 - degrees)) + centroid[0], 2), round(point[1] * math.cos(math.radians(360 - degrees)) + point[0] * math.sin(math.radians(360 - degrees)) + centroid[1], 2))