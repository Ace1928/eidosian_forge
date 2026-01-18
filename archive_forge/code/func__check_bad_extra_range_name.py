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
@error(BAD_EXTRA_RANGE_NAME)
def _check_bad_extra_range_name(self) -> str | None:
    msg: str = ''
    valid = {f'{axis}_name': {'default', *getattr(self, f'extra_{axis}s')} for axis in ('x_range', 'y_range')}
    for place in [*list(Place), 'renderers']:
        for ref in getattr(self, place):
            bad = ', '.join((f"{axis}='{getattr(ref, axis)}'" for axis, keys in valid.items() if getattr(ref, axis, 'default') not in keys))
            if bad:
                msg += (', ' if msg else '') + f'{bad} [{ref}]'
    if msg:
        return msg