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
class HelpGTK4(backend_tools.ToolHelpBase):

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

    def _is_valid_shortcut(self, key):
        """
        Check for a valid shortcut to be displayed.

        - GTK will never send 'cmd+' (see `FigureCanvasGTK4._get_key`).
        - The shortcut window only shows keyboard shortcuts, not mouse buttons.
        """
        return 'cmd+' not in key and (not key.startswith('MouseButton.'))

    def trigger(self, *args):
        section = Gtk.ShortcutsSection()
        for name, tool in sorted(self.toolmanager.tools.items()):
            if not tool.description:
                continue
            group = Gtk.ShortcutsGroup()
            section.append(group)
            child = group.get_first_child()
            while child is not None:
                child.set_visible(False)
                child = child.get_next_sibling()
            shortcut = Gtk.ShortcutsShortcut(accelerator=' '.join((self._normalize_shortcut(key) for key in self.toolmanager.get_tool_keymap(name) if self._is_valid_shortcut(key))), title=tool.name, subtitle=tool.description)
            group.append(shortcut)
        window = Gtk.ShortcutsWindow(title='Help', modal=True, transient_for=self._figure.canvas.get_root())
        window.set_child(section)
        window.show()