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
class Fast(Design):
    modifiers = {Accordion: {'active_header_background': 'var(--neutral-fill-active)'}, Tabulator: {'theme': 'fast'}, Viewable: {'stylesheets': [Inherit, f'{CDN_DIST}bundled/theme/fast.css']}}
    _resources = {'font': {'opensans': f'https:{FONT_URL}'}, 'js_modules': {'fast': f'{config.npm_cdn}/@microsoft/fast-components@2.30.6/dist/fast-components.js', 'fast-design': 'js/fast_design.js'}, 'bundle': True, 'tarball': {'fast': {'tar': 'https://registry.npmjs.org/@microsoft/fast-components/-/fast-components-2.30.6.tgz', 'src': 'package/', 'dest': '@microsoft/fast-components@2.30.6', 'exclude': ['*.d.ts', '*.json', '*.md', '*/esm/*']}}}
    _themes = {'default': FastDefaultTheme, 'dark': FastDarkTheme}

    def _wrapper(self, model):
        return FastWrapper(design=None, object=model, style=self.theme.style)