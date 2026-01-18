import sys
from inspect import getmro
import param
from panel.layout import Row, Tabs
from panel.pane import PaneBase
from panel.util import param_name
from .core import DynamicMap, Element, HoloMap, Layout, Overlay, Store, ViewableElement
from .core.util import isscalar
from .element import Curve, Path, Points, Polygons, Rectangles, Table
from .plotting.links import (
from .streams import BoxEdit, CurveEdit, PointDraw, PolyDraw, PolyEdit, Selection1D
class CurveAnnotator(_GeomAnnotator):
    """
    Annotator which allows editing a Curve element and associating values
    with each vertex using a Table.
    """
    default_opts = param.Dict(default={'responsive': True, 'min_height': 400, 'padding': 0.1, 'framewise': True}, doc='\n        Opts to apply to the element.')
    object = param.ClassSelector(class_=Curve, doc='\n        Points element to edit and annotate.')
    vertex_style = param.Dict(default={'size': 10}, doc='\n        Options to apply to vertices during drawing and editing.')
    _stream_type = CurveEdit

    def _init_stream(self):
        name = param_name(self.name)
        self._stream = self._stream_type(source=self.plot, data={}, tooltip=f'{name} Tool', style=self.vertex_style)