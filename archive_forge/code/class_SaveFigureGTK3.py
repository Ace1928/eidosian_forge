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
@backend_tools._register_tool_class(FigureCanvasGTK3)
class SaveFigureGTK3(backend_tools.SaveFigureBase):

    def trigger(self, *args, **kwargs):
        NavigationToolbar2GTK3.save_figure(self._make_classic_style_pseudo_toolbar())