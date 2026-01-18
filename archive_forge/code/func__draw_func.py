import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def _draw_func(self, drawing_area, ctx, width, height):
    self.on_draw_event(self, ctx)
    self._post_draw(self, ctx)