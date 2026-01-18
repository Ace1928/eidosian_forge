from __future__ import annotations
import functools
import os
import pathlib
from typing import (
import param
from bokeh.models import ImportedStyleSheet
from bokeh.themes import Theme as _BkTheme, _dark_minimal, built_in_themes
from ..config import config
from ..io.resources import (
from ..util import relative_to
class DarkTheme(Theme):
    """
    Baseclass for dark themes.
    """
    base_css = param.Filename(default=THEME_CSS / 'dark.css')
    bokeh_theme = param.ClassSelector(class_=(_BkTheme, str), default=_BkTheme(json=BOKEH_DARK))
    _name: ClassVar[str] = 'dark'