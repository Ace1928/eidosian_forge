import functools
import logging
import os
from pathlib import Path
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, GObject, Gtk, Gdk
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
@_BackendGTK.export
class _BackendGTK3(_BackendGTK):
    FigureCanvas = FigureCanvasGTK3
    FigureManager = FigureManagerGTK3