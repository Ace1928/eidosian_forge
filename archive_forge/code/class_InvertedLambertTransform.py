import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.axes import Axes
import matplotlib.axis as maxis
from matplotlib.patches import Circle
from matplotlib.path import Path
import matplotlib.spines as mspines
from matplotlib.ticker import (
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform
class InvertedLambertTransform(_GeoTransform):

    def __init__(self, center_longitude, center_latitude, resolution):
        _GeoTransform.__init__(self, resolution)
        self._center_longitude = center_longitude
        self._center_latitude = center_latitude

    @_api.rename_parameter('3.8', 'xy', 'values')
    def transform_non_affine(self, values):
        x, y = values.T
        clong = self._center_longitude
        clat = self._center_latitude
        p = np.maximum(np.hypot(x, y), 1e-09)
        c = 2 * np.arcsin(0.5 * p)
        sin_c = np.sin(c)
        cos_c = np.cos(c)
        latitude = np.arcsin(cos_c * np.sin(clat) + y * sin_c * np.cos(clat) / p)
        longitude = clong + np.arctan(x * sin_c / (p * np.cos(clat) * cos_c - y * np.sin(clat) * sin_c))
        return np.column_stack([longitude, latitude])

    def inverted(self):
        return LambertAxes.LambertTransform(self._center_longitude, self._center_latitude, self._resolution)