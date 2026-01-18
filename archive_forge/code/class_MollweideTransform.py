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
class MollweideTransform(_GeoTransform):
    """The base Mollweide transform."""

    @_api.rename_parameter('3.8', 'll', 'values')
    def transform_non_affine(self, values):

        def d(theta):
            delta = -(theta + np.sin(theta) - pi_sin_l) / (1 + np.cos(theta))
            return (delta, np.abs(delta) > 0.001)
        longitude, latitude = values.T
        clat = np.pi / 2 - np.abs(latitude)
        ihigh = clat < 0.087
        ilow = ~ihigh
        aux = np.empty(latitude.shape, dtype=float)
        if ilow.any():
            pi_sin_l = np.pi * np.sin(latitude[ilow])
            theta = 2.0 * latitude[ilow]
            delta, large_delta = d(theta)
            while np.any(large_delta):
                theta[large_delta] += delta[large_delta]
                delta, large_delta = d(theta)
            aux[ilow] = theta / 2
        if ihigh.any():
            e = clat[ihigh]
            d = 0.5 * (3 * np.pi * e ** 2) ** (1.0 / 3)
            aux[ihigh] = (np.pi / 2 - d) * np.sign(latitude[ihigh])
        xy = np.empty(values.shape, dtype=float)
        xy[:, 0] = 2.0 * np.sqrt(2.0) / np.pi * longitude * np.cos(aux)
        xy[:, 1] = np.sqrt(2.0) * np.sin(aux)
        return xy

    def inverted(self):
        return MollweideAxes.InvertedMollweideTransform(self._resolution)