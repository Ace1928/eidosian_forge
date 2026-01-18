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
def _show_shortcuts_window(self):
    section = Gtk.ShortcutsSection()
    for name, tool in sorted(self.toolmanager.tools.items()):
        if not tool.description:
            continue
        group = Gtk.ShortcutsGroup()
        section.add(group)
        group.forall(lambda widget, data: widget.set_visible(False), None)
        shortcut = Gtk.ShortcutsShortcut(accelerator=' '.join((self._normalize_shortcut(key) for key in self.toolmanager.get_tool_keymap(name) if self._is_valid_shortcut(key))), title=tool.name, subtitle=tool.description)
        group.add(shortcut)
    window = Gtk.ShortcutsWindow(title='Help', modal=True, transient_for=self._figure.canvas.get_toplevel())
    section.show()
    window.add(section)
    window.show_all()