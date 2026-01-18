import itertools
import operator
import warnings
import matplotlib
import matplotlib.artist
import matplotlib.collections as mcollections
import matplotlib.text
import matplotlib.ticker as mticker
import matplotlib.transforms as mtrans
import numpy as np
import shapely.geometry as sgeom
import cartopy
from cartopy.crs import PlateCarree, Projection, _RectangularProjection
from cartopy.mpl.ticker import (
def _axes_domain(self, nx=None, ny=None):
    """Return lon_range, lat_range"""
    DEBUG = False
    transform = self._crs_transform()
    ax_transform = self.axes.transAxes
    desired_trans = ax_transform - transform
    nx = nx or 100
    ny = ny or 100
    x = np.linspace(1e-09, 1 - 1e-09, nx)
    y = np.linspace(1e-09, 1 - 1e-09, ny)
    x, y = np.meshgrid(x, y)
    coords = np.column_stack((x.ravel(), y.ravel()))
    in_data = desired_trans.transform(coords)
    ax_to_bkg_patch = self.axes.transAxes - self.axes.patch.get_transform()
    background_coord = ax_to_bkg_patch.transform(coords)
    ok = self.axes.patch.get_path().contains_points(background_coord)
    if DEBUG:
        import matplotlib.pyplot as plt
        plt.plot(coords[ok, 0], coords[ok, 1], 'or', clip_on=False, transform=ax_transform)
        plt.plot(coords[~ok, 0], coords[~ok, 1], 'ob', clip_on=False, transform=ax_transform)
    inside = in_data[ok, :]
    if inside.size == 0:
        lon_range = self.crs.x_limits
        lat_range = self.crs.y_limits
    else:
        lat_max = np.compress(np.isfinite(inside[:, 1]), inside[:, 1])
        if lat_max.size == 0:
            lon_range = self.crs.x_limits
            lat_range = self.crs.y_limits
        else:
            lat_max = lat_max.max()
            lon_range = (np.nanmin(inside[:, 0]), np.nanmax(inside[:, 0]))
            lat_range = (np.nanmin(inside[:, 1]), lat_max)
    crs = self.crs
    if isinstance(crs, Projection):
        lon_range = np.clip(lon_range, *crs.x_limits)
        lat_range = np.clip(lat_range, *crs.y_limits)
        prct = np.abs(np.diff(lon_range) / np.diff(crs.x_limits))
        if prct > 0.9:
            lon_range = crs.x_limits
    if self.xlim is not None:
        if np.iterable(self.xlim):
            lon_range = self.xlim
        else:
            lon_range = (-self.xlim, self.xlim)
    if self.ylim is not None:
        if np.iterable(self.ylim):
            lat_range = self.ylim
        else:
            lat_range = (-self.ylim, self.ylim)
    return (lon_range, lat_range)