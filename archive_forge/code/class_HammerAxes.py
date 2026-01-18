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
class HammerAxes(GeoAxes):
    name = 'hammer'

    class HammerTransform(_GeoTransform):
        """The base Hammer transform."""

        @_api.rename_parameter('3.8', 'll', 'values')
        def transform_non_affine(self, values):
            longitude, latitude = values.T
            half_long = longitude / 2.0
            cos_latitude = np.cos(latitude)
            sqrt2 = np.sqrt(2.0)
            alpha = np.sqrt(1.0 + cos_latitude * np.cos(half_long))
            x = 2.0 * sqrt2 * (cos_latitude * np.sin(half_long)) / alpha
            y = sqrt2 * np.sin(latitude) / alpha
            return np.column_stack([x, y])

        def inverted(self):
            return HammerAxes.InvertedHammerTransform(self._resolution)

    class InvertedHammerTransform(_GeoTransform):

        @_api.rename_parameter('3.8', 'xy', 'values')
        def transform_non_affine(self, values):
            x, y = values.T
            z = np.sqrt(1 - (x / 4) ** 2 - (y / 2) ** 2)
            longitude = 2 * np.arctan(z * x / (2 * (2 * z ** 2 - 1)))
            latitude = np.arcsin(y * z)
            return np.column_stack([longitude, latitude])

        def inverted(self):
            return HammerAxes.HammerTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.clear()

    def _get_core_transform(self, resolution):
        return self.HammerTransform(resolution)