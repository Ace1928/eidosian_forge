import collections
import contextlib
import functools
import json
import os
from pathlib import Path
import warnings
import weakref
import matplotlib as mpl
import matplotlib.artist
import matplotlib.axes
import matplotlib.contour
from matplotlib.image import imread
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
import numpy as np
import numpy.ma as ma
import shapely.geometry as sgeom
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl import _MPL_38
import cartopy.mpl.contour
import cartopy.mpl.feature_artist as feature_artist
import cartopy.mpl.geocollection
import cartopy.mpl.patch as cpatch
from cartopy.mpl.slippy_image_artist import SlippyImageArtist
def _get_extent_geom(self, crs=None):
    with self.hold_limits():
        if self.get_autoscale_on():
            self.autoscale_view()
        [x1, y1], [x2, y2] = self.viewLim.get_points()
    domain_in_src_proj = sgeom.Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
    if crs is None:
        proj = self.projection
    elif isinstance(crs, ccrs.Projection):
        proj = crs
    elif isinstance(crs, ccrs.RotatedGeodetic):
        proj = ccrs.RotatedPole(crs.proj4_params['lon_0'] - 180, crs.proj4_params['o_lat_p'])
        warnings.warn(f'Approximating coordinate system {crs!r} with a RotatedPole projection.')
    elif hasattr(crs, 'is_geodetic') and crs.is_geodetic():
        proj = ccrs.PlateCarree(globe=crs.globe)
        warnings.warn(f'Approximating coordinate system {crs!r} with the PlateCarree projection.')
    else:
        raise ValueError(f'Cannot determine extent in coordinate system {crs!r}')
    boundary_poly = sgeom.Polygon(self.projection.boundary)
    if proj != self.projection:
        eroded_boundary = boundary_poly.buffer(-self.projection.threshold)
        geom_in_src_proj = eroded_boundary.intersection(domain_in_src_proj)
        geom_in_crs = proj.project_geometry(geom_in_src_proj, self.projection)
    else:
        geom_in_crs = boundary_poly.intersection(domain_in_src_proj)
    return geom_in_crs