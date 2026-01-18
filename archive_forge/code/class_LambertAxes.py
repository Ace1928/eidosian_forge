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
class LambertAxes(GeoAxes):
    name = 'lambert'

    class LambertTransform(_GeoTransform):
        """The base Lambert transform."""

        def __init__(self, center_longitude, center_latitude, resolution):
            """
            Create a new Lambert transform.  Resolution is the number of steps
            to interpolate between each input line segment to approximate its
            path in curved Lambert space.
            """
            _GeoTransform.__init__(self, resolution)
            self._center_longitude = center_longitude
            self._center_latitude = center_latitude

        @_api.rename_parameter('3.8', 'll', 'values')
        def transform_non_affine(self, values):
            longitude, latitude = values.T
            clong = self._center_longitude
            clat = self._center_latitude
            cos_lat = np.cos(latitude)
            sin_lat = np.sin(latitude)
            diff_long = longitude - clong
            cos_diff_long = np.cos(diff_long)
            inner_k = np.maximum(1 + np.sin(clat) * sin_lat + np.cos(clat) * cos_lat * cos_diff_long, 1e-15)
            k = np.sqrt(2 / inner_k)
            x = k * cos_lat * np.sin(diff_long)
            y = k * (np.cos(clat) * sin_lat - np.sin(clat) * cos_lat * cos_diff_long)
            return np.column_stack([x, y])

        def inverted(self):
            return LambertAxes.InvertedLambertTransform(self._center_longitude, self._center_latitude, self._resolution)

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

    def __init__(self, *args, center_longitude=0, center_latitude=0, **kwargs):
        self._longitude_cap = np.pi / 2
        self._center_longitude = center_longitude
        self._center_latitude = center_latitude
        super().__init__(*args, **kwargs)
        self.set_aspect('equal', adjustable='box', anchor='C')
        self.clear()

    def clear(self):
        super().clear()
        self.yaxis.set_major_formatter(NullFormatter())

    def _get_core_transform(self, resolution):
        return self.LambertTransform(self._center_longitude, self._center_latitude, resolution)

    def _get_affine_transform(self):
        return Affine2D().scale(0.25).translate(0.5, 0.5)