from __future__ import annotations
import logging # isort:skip
from contextlib import contextmanager
from typing import (
import xyzservices
from ..core.enums import (
from ..core.properties import (
from ..core.property.struct import Optional
from ..core.property_mixins import ScalarFillProps, ScalarLineProps
from ..core.query import find
from ..core.validation import error, warning
from ..core.validation.errors import (
from ..core.validation.warnings import MISSING_RENDERERS
from ..model import Model
from ..util.strings import nice_join
from ..util.warnings import warn
from .annotations import Annotation, Legend, Title
from .axes import Axis
from .dom import HTML
from .glyphs import Glyph
from .grids import Grid
from .layouts import GridCommon, LayoutDOM
from .ranges import (
from .renderers import GlyphRenderer, Renderer, TileRenderer
from .scales import (
from .sources import ColumnarDataSource, ColumnDataSource, DataSource
from .tiles import TileSource, WMTSTileSource
from .tools import HoverTool, Tool, Toolbar
class _list_attr_splat(list):

    def __setattr__(self, attr, value):
        for x in self:
            setattr(x, attr, value)

    def __getattribute__(self, attr):
        if attr in dir(list):
            return list.__getattribute__(self, attr)
        if len(self) == 0:
            raise AttributeError("Trying to access %r attribute on an empty 'splattable' list" % attr)
        if len(self) == 1:
            return getattr(self[0], attr)
        try:
            return _list_attr_splat([getattr(x, attr) for x in self])
        except Exception:
            raise AttributeError(f"Trying to access {attr!r} attribute on a 'splattable' list, but list items have no {attr!r} attribute")

    def __dir__(self):
        if len({type(x) for x in self}) == 1:
            return dir(self[0])
        else:
            return dir(self)