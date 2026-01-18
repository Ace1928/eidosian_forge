import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def _is_valid_shortcut(self, key):
    """
        Check for a valid shortcut to be displayed.

        - GTK will never send 'cmd+' (see `FigureCanvasGTK4._get_key`).
        - The shortcut window only shows keyboard shortcuts, not mouse buttons.
        """
    return 'cmd+' not in key and (not key.startswith('MouseButton.'))