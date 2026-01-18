import itertools
import time
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
class FakeProjection(ccrs.PlateCarree):

    def __init__(self, left_offset=0, right_offset=0):
        self.left_offset = left_offset
        self.right_offset = right_offset
        self._half_width = 180
        self._half_height = 90
        ccrs.PlateCarree.__init__(self)

    @property
    def boundary(self):
        w, h = (self._half_width, self._half_height)
        return sgeom.LineString([(-w + self.left_offset, -h), (-w + self.left_offset, h), (w - self.right_offset, h), (w - self.right_offset, -h), (-w + self.left_offset, -h)])