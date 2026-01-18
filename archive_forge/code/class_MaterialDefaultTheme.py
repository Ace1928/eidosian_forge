from __future__ import annotations
import pathlib
import param
from bokeh.themes import Theme as _BkTheme
from ..config import config
from ..io.resources import CDN_DIST
from ..layout import Accordion, Card
from ..viewable import Viewable
from ..widgets import Tabulator
from ..widgets.indicators import Dial, Number, String
from .base import (
class MaterialDefaultTheme(MaterialThemeMixin, DefaultTheme):
    """
    The MaterialDefaultTheme is a light theme.
    """
    bokeh_theme = param.ClassSelector(class_=(_BkTheme, str), default=_BkTheme(json=MATERIAL_THEME))