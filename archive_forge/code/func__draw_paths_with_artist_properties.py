import functools
import inspect
import math
from numbers import Number, Real
import textwrap
from types import SimpleNamespace
from collections import namedtuple
from matplotlib.transforms import Affine2D
import numpy as np
import matplotlib as mpl
from . import (_api, artist, cbook, colors, _docstring, hatch as mhatch,
from .bezier import (
from .path import Path
from ._enums import JoinStyle, CapStyle
def _draw_paths_with_artist_properties(self, renderer, draw_path_args_list):
    """
        ``draw()`` helper factored out for sharing with `FancyArrowPatch`.

        Configure *renderer* and the associated graphics context *gc*
        from the artist properties, then repeatedly call
        ``renderer.draw_path(gc, *draw_path_args)`` for each tuple
        *draw_path_args* in *draw_path_args_list*.
        """
    renderer.open_group('patch', self.get_gid())
    gc = renderer.new_gc()
    gc.set_foreground(self._edgecolor, isRGBA=True)
    lw = self._linewidth
    if self._edgecolor[3] == 0 or self._linestyle == 'None':
        lw = 0
    gc.set_linewidth(lw)
    gc.set_dashes(*self._dash_pattern)
    gc.set_capstyle(self._capstyle)
    gc.set_joinstyle(self._joinstyle)
    gc.set_antialiased(self._antialiased)
    self._set_gc_clip(gc)
    gc.set_url(self._url)
    gc.set_snap(self.get_snap())
    gc.set_alpha(self._alpha)
    if self._hatch:
        gc.set_hatch(self._hatch)
        gc.set_hatch_color(self._hatch_color)
    if self.get_sketch_params() is not None:
        gc.set_sketch_params(*self.get_sketch_params())
    if self.get_path_effects():
        from matplotlib.patheffects import PathEffectRenderer
        renderer = PathEffectRenderer(self.get_path_effects(), renderer)
    for draw_path_args in draw_path_args_list:
        renderer.draw_path(gc, *draw_path_args)
    gc.restore()
    renderer.close_group('patch')
    self.stale = False