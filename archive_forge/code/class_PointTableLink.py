import param
from holoviews.plotting.links import Link, RectanglesTableLink as HvRectanglesTableLink
from holoviews.plotting.bokeh.links import (
from holoviews.core.util import dimension_sanitizer
class PointTableLink(Link):
    """
    Defines a Link between a Points type and a Table which will
    display the projected coordinates.
    """
    point_columns = param.List(default=[])
    _requires_target = True

    def __init__(self, source, target, **params):
        if 'point_columns' not in params:
            dimensions = [dimension_sanitizer(d.name) for d in target.dimensions()[:2]]
            params['point_columns'] = dimensions
        super().__init__(source, target, **params)