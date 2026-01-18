import re
import os
from ast import literal_eval
from functools import partial
from copy import copy
from kivy import kivy_data_dir
from kivy.config import Config
from kivy.utils import platform
from kivy.graphics.texture import Texture
from kivy.core import core_select_lib
from kivy.core.text.text_layout import layout_text, LayoutWord
from kivy.resources import resource_find, resource_add_path
from kivy.compat import PY2
from kivy.setupconfig import USE_SDL2, USE_PANGOFT2
from kivy.logger import Logger
def _render_real(self):
    lines = self._cached_lines
    options = self._default_line_options(lines)
    if options is None:
        return self.clear_texture()
    old_opts = self.options
    ih = self._internal_size[1]
    size = self.size
    valign = options['valign']
    padding_top = options['padding'][1]
    if valign == 'bottom':
        y = int(size[1] - ih + padding_top)
    elif valign == 'top':
        y = int(padding_top)
    elif valign in ('middle', 'center'):
        y = int((size[1] - ih + 2 * padding_top) / 2)
    self._render_begin()
    self.render_lines(lines, options, self._render_text, y, size)
    data = self._render_end()
    assert data
    self.options = old_opts
    if data is not None and data.width > 1:
        self.texture.blit_data(data)