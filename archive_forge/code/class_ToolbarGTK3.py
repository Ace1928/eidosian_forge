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
class ToolbarGTK3(ToolContainerBase, Gtk.Box):
    _icon_extension = '-symbolic.svg'

    def __init__(self, toolmanager):
        ToolContainerBase.__init__(self, toolmanager)
        Gtk.Box.__init__(self)
        self.set_property('orientation', Gtk.Orientation.HORIZONTAL)
        self._message = Gtk.Label()
        self._message.set_justify(Gtk.Justification.RIGHT)
        self.pack_end(self._message, False, False, 0)
        self.show_all()
        self._groups = {}
        self._toolitems = {}

    def add_toolitem(self, name, group, position, image_file, description, toggle):
        if toggle:
            button = Gtk.ToggleToolButton()
        else:
            button = Gtk.ToolButton()
        button.set_label(name)
        if image_file is not None:
            image = Gtk.Image.new_from_gicon(Gio.Icon.new_for_string(image_file), Gtk.IconSize.LARGE_TOOLBAR)
            button.set_icon_widget(image)
        if position is None:
            position = -1
        self._add_button(button, group, position)
        signal = button.connect('clicked', self._call_tool, name)
        button.set_tooltip_text(description)
        button.show_all()
        self._toolitems.setdefault(name, [])
        self._toolitems[name].append((button, signal))

    def _add_button(self, button, group, position):
        if group not in self._groups:
            if self._groups:
                self._add_separator()
            toolbar = Gtk.Toolbar()
            toolbar.set_style(Gtk.ToolbarStyle.ICONS)
            self.pack_start(toolbar, False, False, 0)
            toolbar.show_all()
            self._groups[group] = toolbar
        self._groups[group].insert(button, position)

    def _call_tool(self, btn, name):
        self.trigger_tool(name)

    def toggle_toolitem(self, name, toggled):
        if name not in self._toolitems:
            return
        for toolitem, signal in self._toolitems[name]:
            toolitem.handler_block(signal)
            toolitem.set_active(toggled)
            toolitem.handler_unblock(signal)

    def remove_toolitem(self, name):
        if name not in self._toolitems:
            self.toolmanager.message_event(f'{name} not in toolbar', self)
            return
        for group in self._groups:
            for toolitem, _signal in self._toolitems[name]:
                if toolitem in self._groups[group]:
                    self._groups[group].remove(toolitem)
        del self._toolitems[name]

    def _add_separator(self):
        sep = Gtk.Separator()
        sep.set_property('orientation', Gtk.Orientation.VERTICAL)
        self.pack_start(sep, False, True, 0)
        sep.show_all()

    def set_message(self, s):
        self._message.set_label(s)