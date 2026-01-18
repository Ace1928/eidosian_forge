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
@error(INCOMPATIBLE_SCALE_AND_RANGE)
def _check_compatible_scale_and_ranges(self) -> str | None:
    incompatible: list[str] = []
    x_ranges = list(self.extra_x_ranges.values())
    if self.x_range:
        x_ranges.append(self.x_range)
    y_ranges = list(self.extra_y_ranges.values())
    if self.y_range:
        y_ranges.append(self.y_range)
    if self.x_scale is not None:
        for rng in x_ranges:
            if isinstance(rng, (DataRange1d, Range1d)) and (not isinstance(self.x_scale, (LinearScale, LogScale))):
                incompatible.append(f'incompatibility on x-dimension: {rng}, {self.x_scale}')
            elif isinstance(rng, FactorRange) and (not isinstance(self.x_scale, CategoricalScale)):
                incompatible.append(f'incompatibility on x-dimension: {rng}, {self.x_scale}')
    if self.y_scale is not None:
        for rng in y_ranges:
            if isinstance(rng, (DataRange1d, Range1d)) and (not isinstance(self.y_scale, (LinearScale, LogScale))):
                incompatible.append(f'incompatibility on y-dimension: {rng}, {self.y_scale}')
            elif isinstance(rng, FactorRange) and (not isinstance(self.y_scale, CategoricalScale)):
                incompatible.append(f'incompatibility on y-dimension: {rng}, {self.y_scale}')
    if incompatible:
        return ', '.join(incompatible) + ' [%s]' % self