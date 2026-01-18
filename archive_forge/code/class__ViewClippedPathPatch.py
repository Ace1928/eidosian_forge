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
class _ViewClippedPathPatch(mpatches.PathPatch):

    def __init__(self, axes, **kwargs):
        self._original_path = mpath.Path(np.empty((0, 2)))
        super().__init__(self._original_path, **kwargs)
        self._axes = axes
        self._trans_wrap = mtransforms.TransformWrapper(self.get_transform())

    def set_transform(self, transform):
        self._trans_wrap.set(transform)
        super().set_transform(self._trans_wrap)

    def set_boundary(self, path, transform):
        self._original_path = path
        self.set_transform(transform)
        self.stale = True

    def set_path(self, path):
        self._path = path

    def _adjust_location(self):
        if self.stale:
            self.set_path(self._original_path.clip_to_bbox(self.axes.viewLim))
            self._trans_wrap.invalidate()

    @matplotlib.artist.allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        self._adjust_location()
        super().draw(renderer, *args, **kwargs)