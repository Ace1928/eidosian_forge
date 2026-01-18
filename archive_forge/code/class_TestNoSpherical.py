from unittest import mock
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot, InterProjectionTransform
class TestNoSpherical:

    def setup_method(self):
        self.ax = plt.axes(projection=ccrs.PlateCarree())
        self.data = np.arange(12).reshape((3, 4))

    def test_contour(self):
        with pytest.raises(ValueError):
            self.ax.contour(self.data, transform=ccrs.Geodetic())

    def test_contourf(self):
        with pytest.raises(ValueError):
            self.ax.contourf(self.data, transform=ccrs.Geodetic())

    def test_pcolor(self):
        with pytest.raises(ValueError):
            self.ax.pcolor(self.data, transform=ccrs.Geodetic())

    def test_pcolormesh(self):
        with pytest.raises(ValueError):
            self.ax.pcolormesh(self.data, transform=ccrs.Geodetic())