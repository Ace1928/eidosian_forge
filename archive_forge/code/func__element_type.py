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
@property
def _element_type(self):
    return self.param.object.class_