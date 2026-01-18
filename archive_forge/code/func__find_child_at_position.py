import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def _find_child_at_position(self, group, position):
    children = [None]
    child = self._groups[group].get_first_child()
    while child is not None:
        children.append(child)
        child = child.get_next_sibling()
    return children[position]