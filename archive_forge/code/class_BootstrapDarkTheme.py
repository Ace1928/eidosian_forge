from __future__ import annotations
import pathlib
import param
from ..io.resources import CDN_DIST, CSS_URLS, JS_URLS
from ..layout import Accordion, Card
from ..viewable import Viewable
from ..widgets import Number, Tabulator
from .base import (
class BootstrapDarkTheme(DarkTheme):
    """
    The BootstrapDarkTheme is a Dark Theme in the style of Bootstrap
    """
    css = param.Filename(default=pathlib.Path(__file__).parent / 'css' / 'bootstrap_dark.css')
    _bs_theme = 'dark'
    modifiers = {Number: {'default_color': 'white'}}