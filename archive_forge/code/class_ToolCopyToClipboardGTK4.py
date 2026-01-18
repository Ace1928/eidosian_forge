import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
@backend_tools._register_tool_class(FigureCanvasGTK4)
class ToolCopyToClipboardGTK4(backend_tools.ToolCopyToClipboardBase):

    def trigger(self, *args, **kwargs):
        with io.BytesIO() as f:
            self.canvas.print_rgba(f)
            w, h = self.canvas.get_width_height()
            pb = GdkPixbuf.Pixbuf.new_from_data(f.getbuffer(), GdkPixbuf.Colorspace.RGB, True, 8, w, h, w * 4)
        clipboard = self.canvas.get_clipboard()
        clipboard.set(pb)