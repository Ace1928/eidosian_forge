from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
def _prepare_with_copy(geometry):
    """Prepare without modifying inplace"""
    geometry = shapely.transform(geometry, lambda x: x)
    shapely.prepare(geometry)
    return geometry