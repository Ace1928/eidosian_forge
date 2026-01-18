import param
from cartopy.crs import GOOGLE_MERCATOR, PlateCarree, Mercator
from bokeh.models.tools import BoxZoomTool, WheelZoomTool
from bokeh.models import MercatorTickFormatter, MercatorTicker, CustomJSHover
from holoviews.core.dimension import Dimension
from holoviews.core.util import dimension_sanitizer
from holoviews.plotting.bokeh.element import ElementPlot, OverlayPlot as HvOverlayPlot
from ...element import is_geographic, _Element, Shape
from ..plot import ProjectionPlot
def _update_ranges(self, element, ranges):
    super()._update_ranges(element, ranges)
    if not self.geographic:
        return
    if self.fixed_bounds:
        self.handles['x_range'].bounds = self.projection.x_limits
        self.handles['y_range'].bounds = self.projection.y_limits
    if self.projection is GOOGLE_MERCATOR:
        options = self._traverse_options(element, 'plot', ['default_span'], defaults=False)
        min_interval = options['default_span'][0] if options.get('default_span') else 5
        for r in ('x_range', 'y_range'):
            ax_range = self.handles[r]
            start, end = (ax_range.start, ax_range.end)
            if end - start < min_interval:
                mid = (start + end) / 2.0
                ax_range.start = mid - min_interval / 2.0
                ax_range.end = mid + min_interval / 2.0
            ax_range.min_interval = min_interval