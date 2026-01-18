from __future__ import annotations
import pathlib
import param
from ..io.resources import CDN_DIST, CSS_URLS, JS_URLS
from ..layout import Accordion, Card
from ..viewable import Viewable
from ..widgets import Number, Tabulator
from .base import (
class BootstrapDefaultTheme(DefaultTheme):
    """
    The BootstrapDefaultTheme is a light theme.
    """
    css = param.Filename(default=pathlib.Path(__file__).parent / 'css' / 'bootstrap_default.css')
    _bs_theme = 'light'