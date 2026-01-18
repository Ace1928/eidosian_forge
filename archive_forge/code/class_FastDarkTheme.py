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
class FastDarkTheme(DarkTheme):
    style = param.ClassSelector(default=DARK_STYLE, class_=FastStyle)
    modifiers = {Dial: {'label_color': 'white'}, Number: {'default_color': 'var(--neutral-foreground-rest)'}, String: {'default_color': 'var(--neutral-foreground-rest)'}}
    __abstract = True

    @property
    def bokeh_theme(self):
        return _BkTheme(json=self.style.create_bokeh_theme())