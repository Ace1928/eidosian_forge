import logging
import sys
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backend_tools import Cursors
import gi
from gi.repository import Gdk, Gio, GLib, Gtk
class RubberbandGTK(backend_tools.RubberbandBase):

    def draw_rubberband(self, x0, y0, x1, y1):
        _NavigationToolbar2GTK.draw_rubberband(self._make_classic_style_pseudo_toolbar(), None, x0, y0, x1, y1)

    def remove_rubberband(self):
        _NavigationToolbar2GTK.remove_rubberband(self._make_classic_style_pseudo_toolbar())