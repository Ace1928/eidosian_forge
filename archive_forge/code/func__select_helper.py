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
def _select_helper(args, kwargs):
    """ Allow flexible selector syntax.

    Returns:
        dict

    """
    if len(args) > 1:
        raise TypeError('select accepts at most ONE positional argument.')
    if len(args) > 0 and len(kwargs) > 0:
        raise TypeError('select accepts EITHER a positional argument, OR keyword arguments (not both).')
    if len(args) == 0 and len(kwargs) == 0:
        raise TypeError('select requires EITHER a positional argument, OR keyword arguments.')
    if args:
        arg = args[0]
        if isinstance(arg, dict):
            selector = arg
        elif isinstance(arg, str):
            selector = dict(name=arg)
        elif isinstance(arg, type) and issubclass(arg, Model):
            selector = {'type': arg}
        else:
            raise TypeError('selector must be a dictionary, string or plot object.')
    elif 'selector' in kwargs:
        if len(kwargs) == 1:
            selector = kwargs['selector']
        else:
            raise TypeError("when passing 'selector' keyword arg, not other keyword args may be present")
    else:
        selector = kwargs
    return selector