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
class DefaultTheme(Theme):
    """
    Baseclass for default or light themes.
    """
    base_css = param.Filename(default=THEME_CSS / 'default.css')
    _name: ClassVar[str] = 'default'