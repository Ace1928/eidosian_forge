from __future__ import annotations
from ..io.resources import CDN_DIST
from ..viewable import Viewable
from ..widgets.indicators import Number
from .base import (
class NativeDarkTheme(DarkTheme):
    modifiers = {Number: {'default_color': 'white'}}