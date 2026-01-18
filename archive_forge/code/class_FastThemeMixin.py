from __future__ import annotations
import pathlib
import param
from bokeh.themes import Theme as _BkTheme
from ..config import config
from ..io.resources import CDN_DIST
from ..layout import Accordion
from ..reactive import ReactiveHTML
from ..viewable import Viewable
from ..widgets import Tabulator
from ..widgets.indicators import Dial, Number, String
from .base import (
class FastThemeMixin(param.Parameterized):
    css = param.Filename(default=pathlib.Path(__file__).parent / 'css' / 'fast_variables.css')