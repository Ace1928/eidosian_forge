import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def _add_separator(self):
    sep = Gtk.Separator()
    sep.set_property('orientation', Gtk.Orientation.VERTICAL)
    self._tool_box.append(sep)