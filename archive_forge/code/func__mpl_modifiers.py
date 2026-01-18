import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def _mpl_modifiers(self, controller=None):
    if controller is None:
        surface = self.get_native().get_surface()
        is_over, x, y, event_state = surface.get_device_position(self.get_display().get_default_seat().get_pointer())
    else:
        event_state = controller.get_current_event_state()
    mod_table = [('ctrl', Gdk.ModifierType.CONTROL_MASK), ('alt', Gdk.ModifierType.ALT_MASK), ('shift', Gdk.ModifierType.SHIFT_MASK), ('super', Gdk.ModifierType.SUPER_MASK)]
    return [name for name, mask in mod_table if event_state & mask]