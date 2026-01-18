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
class AitoffAxes(GeoAxes):
    name = 'aitoff'

    class AitoffTransform(_GeoTransform):
        """The base Aitoff transform."""

        @_api.rename_parameter('3.8', 'll', 'values')
        def transform_non_affine(self, values):
            longitude, latitude = values.T
            half_long = longitude / 2.0
            cos_latitude = np.cos(latitude)
            alpha = np.arccos(cos_latitude * np.cos(half_long))
            sinc_alpha = np.sinc(alpha / np.pi)
            x = cos_latitude * np.sin(half_long) / sinc_alpha
            y = np.sin(latitude) / sinc_alpha
            return np.column_stack([x, y])

        def inverted(self):
            return AitoffAxes.InvertedAitoffTransform(self._resolution)

    class InvertedAitoffTransform(_GeoTransform):

        @_api.rename_parameter('3.8', 'xy', 'values')
        def transform_non_affine(self, values):
            return np.full_like(values, np.nan)

        def inverted(self):
            return AitoffAxes.AitoffTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.clear()

    def _get_core_transform(self, resolution):
        return self.AitoffTransform(resolution)