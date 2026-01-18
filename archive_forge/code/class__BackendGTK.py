import logging
import sys
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backend_tools import Cursors
import gi
from gi.repository import Gdk, Gio, GLib, Gtk
class _BackendGTK(_Backend):
    backend_version = '{}.{}.{}'.format(Gtk.get_major_version(), Gtk.get_minor_version(), Gtk.get_micro_version())
    mainloop = _FigureManagerGTK.start_main_loop