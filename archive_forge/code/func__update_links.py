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
def _update_links(self):
    super()._update_links()
    if hasattr(self, '_vertex_link'):
        self._vertex_link.unlink()
    self._vertex_link = self._vertex_table_link(self.plot, self._vertex_table)