from __future__ import annotations
import logging # isort:skip
from ..models import (
from ._figure import BaseFigureOptions
from ._plot import _get_num_minor_ticks
from ._tools import process_active_tools, process_tools_arg
from .glyph_api import GlyphAPI
class GMapFigureOptions(BaseFigureOptions):
    pass