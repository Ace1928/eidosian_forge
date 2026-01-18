import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def _post_draw(self, widget, ctx):
    if self._rubberband_rect is None:
        return
    lw = 1
    dash = 3
    if not self._context_is_scaled:
        x0, y0, w, h = (dim / self.device_pixel_ratio for dim in self._rubberband_rect)
    else:
        x0, y0, w, h = self._rubberband_rect
        lw *= self.device_pixel_ratio
        dash *= self.device_pixel_ratio
    x1 = x0 + w
    y1 = y0 + h
    ctx.move_to(x0, y0)
    ctx.line_to(x0, y1)
    ctx.move_to(x0, y0)
    ctx.line_to(x1, y0)
    ctx.move_to(x0, y1)
    ctx.line_to(x1, y1)
    ctx.move_to(x1, y0)
    ctx.line_to(x1, y1)
    ctx.set_antialias(1)
    ctx.set_line_width(lw)
    ctx.set_dash((dash, dash), 0)
    ctx.set_source_rgb(0, 0, 0)
    ctx.stroke_preserve()
    ctx.set_dash((dash, dash), dash)
    ctx.set_source_rgb(1, 1, 1)
    ctx.stroke()