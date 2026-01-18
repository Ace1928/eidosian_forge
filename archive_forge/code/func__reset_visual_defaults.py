import functools
import logging
import math
from numbers import Real
import weakref
import numpy as np
import matplotlib as mpl
from . import _api, artist, cbook, _docstring
from .artist import Artist
from .font_manager import FontProperties
from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from .textpath import TextPath, TextToPath  # noqa # Logically located here
from .transforms import (
def _reset_visual_defaults(self, text='', color=None, fontproperties=None, usetex=None, parse_math=None, wrap=False, verticalalignment='baseline', horizontalalignment='left', multialignment=None, rotation=None, transform_rotates_text=False, linespacing=None, rotation_mode=None, antialiased=None):
    self.set_text(text)
    self.set_color(mpl._val_or_rc(color, 'text.color'))
    self.set_fontproperties(fontproperties)
    self.set_usetex(usetex)
    self.set_parse_math(mpl._val_or_rc(parse_math, 'text.parse_math'))
    self.set_wrap(wrap)
    self.set_verticalalignment(verticalalignment)
    self.set_horizontalalignment(horizontalalignment)
    self._multialignment = multialignment
    self.set_rotation(rotation)
    self._transform_rotates_text = transform_rotates_text
    self._bbox_patch = None
    self._renderer = None
    if linespacing is None:
        linespacing = 1.2
    self.set_linespacing(linespacing)
    self.set_rotation_mode(rotation_mode)
    self.set_antialiased(antialiased if antialiased is not None else mpl.rcParams['text.antialiased'])