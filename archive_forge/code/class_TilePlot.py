import copy
import param
import numpy as np
from cartopy.crs import GOOGLE_MERCATOR
from bokeh.models import WMTSTileSource, BBoxTileSource, QUADKEYTileSource, SaveTool
from holoviews import Store, Overlay, NdOverlay
from holoviews.core import util
from holoviews.core.options import SkipRendering, Options, Compositor
from holoviews.plotting.bokeh.annotation import TextPlot, LabelsPlot
from holoviews.plotting.bokeh.chart import PointPlot, VectorFieldPlot
from holoviews.plotting.bokeh.geometry import RectanglesPlot, SegmentPlot
from holoviews.plotting.bokeh.graphs import TriMeshPlot, GraphPlot
from holoviews.plotting.bokeh.hex_tiles import hex_binning, HexTilesPlot
from holoviews.plotting.bokeh.path import PolygonPlot, PathPlot, ContourPlot
from holoviews.plotting.bokeh.raster import RasterPlot, RGBPlot, QuadMeshPlot
from ...element import (
from ...operation import (
from ...tile_sources import _ATTRIBUTIONS
from ...util import poly_types, line_types
from .plot import GeoPlot, GeoOverlayPlot
from . import callbacks # noqa
class TilePlot(GeoPlot):
    style_opts = ['alpha', 'render_parents', 'level', 'smoothing', 'min_zoom', 'max_zoom', 'extra_url_vars', 'tile_size', 'use_latlon', 'wrap_around']

    def get_extents(self, element, ranges, range_type='combined', **kwargs):
        extents = super().get_extents(element, ranges, range_type)
        if not self.overlaid and all((e is None or not np.isfinite(e) for e in extents)) and (range_type in ('combined', 'data')):
            (x0, x1), (y0, y1) = (GOOGLE_MERCATOR.x_limits, GOOGLE_MERCATOR.y_limits)
            global_extent = (x0, y0, x1, y1)
            return global_extent
        return extents

    def get_data(self, element, ranges, style):
        if not isinstance(element.data, (str, dict)):
            SkipRendering('WMTS element data must be a URL string, bokeh cannot render %r' % element.data)
        if isinstance(element.data, dict):
            params = {'url': element.data.build_url(scale_factor='@2x'), 'min_zoom': element.data.get('min_zoom', 0), 'max_zoom': element.data.get('max_zoom', 20), 'attribution': element.data.html_attribution}
            tile_source = WMTSTileSource
        else:
            if all((kw in element.data.upper() for kw in ('{X}', '{Y}', '{Z}'))):
                tile_source = WMTSTileSource
            elif '{Q}' in element.data.upper():
                tile_source = QUADKEYTileSource
            elif all((kw in element.data.upper() for kw in ('{XMIN}', '{XMAX}', '{YMIN}', '{YMAX}'))):
                tile_source = BBoxTileSource
            else:
                raise ValueError('Tile source URL format not recognized. Must contain {X}/{Y}/{Z}, {XMIN}/{XMAX}/{YMIN}/{YMAX} or {Q} template strings.')
            params = {'url': element.data}
            for zoom in ('min_zoom', 'max_zoom'):
                if zoom in style:
                    params[zoom] = style[zoom]
            for key, attribution in _ATTRIBUTIONS.items():
                if all((k in element.data for k in key)):
                    params['attribution'] = attribution
        return ({}, {'tile_source': tile_source(**params)}, style)

    def _update_glyph(self, renderer, properties, mapping, glyph, source=None, data=None):
        glyph.url = mapping['tile_source'].url
        glyph.update(**{k: v for k, v in properties.items() if k in glyph.properties()})
        renderer.update(**{k: v for k, v in properties.items() if k in renderer.properties()})

    def _init_glyph(self, plot, mapping, properties):
        """
        Returns a Bokeh glyph object.
        """
        tile_source = mapping['tile_source']
        level = properties.pop('level', 'underlay')
        renderer = plot.add_tile(tile_source, level=level)
        renderer.alpha = properties.get('alpha', 1)
        plot.tools = [t for t in plot.tools if not isinstance(t, SaveTool)]
        return (renderer, tile_source)