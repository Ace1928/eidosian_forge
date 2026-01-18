import weakref
from collections import defaultdict
import param
from ..core.util import dimension_sanitizer
class VertexTableLink(Link):
    """
    Defines a Link between a Path type and a Table that will
    display the vertices of selected path.
    """
    vertex_columns = param.List(default=[])
    _requires_target = True

    def __init__(self, source, target, **params):
        if 'vertex_columns' not in params:
            dimensions = [dimension_sanitizer(d.name) for d in target.dimensions()[:2]]
            params['vertex_columns'] = dimensions
        super().__init__(source, target, **params)