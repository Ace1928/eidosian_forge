import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def _normalize_shortcut(self, key):
    """
        Convert Matplotlib key presses to GTK+ accelerator identifiers.

        Related to `FigureCanvasGTK4._get_key`.
        """
    special = {'backspace': 'BackSpace', 'pagedown': 'Page_Down', 'pageup': 'Page_Up', 'scroll_lock': 'Scroll_Lock'}
    parts = key.split('+')
    mods = ['<' + mod + '>' for mod in parts[:-1]]
    key = parts[-1]
    if key in special:
        key = special[key]
    elif len(key) > 1:
        key = key.capitalize()
    elif key.isupper():
        mods += ['<shift>']
    return ''.join(mods) + key