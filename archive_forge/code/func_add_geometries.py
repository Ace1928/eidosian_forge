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
def add_geometries(self, geoms, crs, **kwargs):
    """
        Add the given shapely geometries (in the given crs) to the axes.

        Parameters
        ----------
        geoms
            A collection of shapely geometries.
        crs
            The cartopy CRS in which the provided geometries are defined.
        styler
            A callable that returns matplotlib patch styling given a geometry.

        Returns
        -------
        A :class:`cartopy.mpl.feature_artist.FeatureArtist` instance
            The instance responsible for drawing the feature.

        Note
        ----
            Matplotlib keyword arguments can be used when drawing the feature.
            This allows standard Matplotlib control over aspects such as
            'facecolor', 'alpha', etc.


        """
    styler = kwargs.pop('styler', None)
    feature = cartopy.feature.ShapelyFeature(geoms, crs, **kwargs)
    return self.add_feature(feature, styler=styler)