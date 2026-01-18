import difflib
from functools import partial
import param
import holoviews as hv
import pandas as pd
import numpy as np
import colorcet as cc
from bokeh.models import HoverTool
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import DynamicMap, HoloMap, Callable
from holoviews.core.overlay import NdOverlay
from holoviews.core.options import Store, Cycle, Palette
from holoviews.core.layout import NdLayout
from holoviews.core.util import max_range
from holoviews.element import (
from holoviews.plotting.bokeh import OverlayPlot, colormap_generator
from holoviews.plotting.util import process_cmap
from holoviews.operation import histogram, apply_when
from holoviews.streams import Buffer, Pipe
from holoviews.util.transform import dim
from packaging.version import Version
from pandas import DatetimeIndex, MultiIndex
from .backend_transforms import _transfer_opts_cur_backend
from .util import (
from .utilities import hvplot_extension
def _apply_layers(self, obj):
    if self.coastline:
        import geoviews as gv
        coastline = gv.feature.coastline.clone()
        if self.coastline in ['10m', '50m', '110m']:
            coastline = coastline.opts(scale=self.coastline)
        elif self.coastline is not True:
            param.main.param.warning("coastline scale of %s not recognized, must be one '10m', '50m' or '110m'." % self.coastline)
        obj = obj * coastline.opts(projection=self.output_projection)
    if self.features:
        import geoviews as gv
        for feature in reversed(self.features):
            feature_obj = getattr(gv.feature, feature)
            if feature_obj is None:
                raise ValueError("Feature %r was not recognized, must be one of 'borders', 'coastline', 'lakes', 'land', 'ocean', 'rivers' and 'states'." % feature)
            feature_obj = feature_obj.clone()
            if isinstance(self.features, dict):
                scale = self.features[feature]
                if scale not in ['10m', '50m', '110m']:
                    param.main.param.warning("Feature scale of %r not recognized, must be one of '10m', '50m' or '110m'." % scale)
                else:
                    feature_obj = feature_obj.opts(scale=scale)
            if feature_obj.group in ['Land', 'Ocean']:
                obj = feature_obj.opts(projection=self.output_projection) * obj
            else:
                obj = obj * feature_obj.opts(projection=self.output_projection)
    if self.tiles and (not self.geo):
        tiles = self._get_tiles(self.tiles, hv.element.tile_sources, hv.element.tiles.Tiles)
        obj = tiles * obj
    elif self.tiles and self.geo:
        import geoviews as gv
        tiles = self._get_tiles(self.tiles, gv.tile_sources.tile_sources, (gv.element.WMTS, hv.element.tiles.Tiles))
        obj = tiles * obj
    return obj