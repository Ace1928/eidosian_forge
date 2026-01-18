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
def add_glyph(self, source_or_glyph: Glyph | ColumnarDataSource, glyph: Glyph | None=None, **kwargs: Any) -> GlyphRenderer:
    """ Adds a glyph to the plot with associated data sources and ranges.

        This function will take care of creating and configuring a Glyph object,
        and then add it to the plot's list of renderers.

        Args:
            source (DataSource) : a data source for the glyphs to all use
            glyph (Glyph) : the glyph to add to the Plot


        Keyword Arguments:
            Any additional keyword arguments are passed on as-is to the
            Glyph initializer.

        Returns:
            GlyphRenderer

        """
    if isinstance(source_or_glyph, ColumnarDataSource):
        source = source_or_glyph
    else:
        source, glyph = (ColumnDataSource(), source_or_glyph)
    if not isinstance(source, DataSource):
        raise ValueError("'source' argument to add_glyph() must be DataSource subclass")
    if not isinstance(glyph, Glyph):
        raise ValueError("'glyph' argument to add_glyph() must be Glyph subclass")
    g = GlyphRenderer(data_source=source, glyph=glyph, **kwargs)
    self.renderers.append(g)
    return g