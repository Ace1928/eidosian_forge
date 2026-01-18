import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
class ToolbarGTK4(ToolContainerBase, Gtk.Box):
    _icon_extension = '-symbolic.svg'

    def __init__(self, toolmanager):
        ToolContainerBase.__init__(self, toolmanager)
        Gtk.Box.__init__(self)
        self.set_property('orientation', Gtk.Orientation.HORIZONTAL)
        self._tool_box = Gtk.Box()
        self.append(self._tool_box)
        self._groups = {}
        self._toolitems = {}
        label = Gtk.Label()
        label.set_markup('<small>\xa0\n\xa0</small>')
        label.set_hexpand(True)
        self.append(label)
        self._message = Gtk.Label()
        self._message.set_justify(Gtk.Justification.RIGHT)
        self.append(self._message)

    def add_toolitem(self, name, group, position, image_file, description, toggle):
        if toggle:
            button = Gtk.ToggleButton()
        else:
            button = Gtk.Button()
        button.set_label(name)
        button.add_css_class('flat')
        if image_file is not None:
            image = Gtk.Image.new_from_gicon(Gio.Icon.new_for_string(image_file))
            button.set_child(image)
            button.add_css_class('image-button')
        if position is None:
            position = -1
        self._add_button(button, group, position)
        signal = button.connect('clicked', self._call_tool, name)
        button.set_tooltip_text(description)
        self._toolitems.setdefault(name, [])
        self._toolitems[name].append((button, signal))

    def _find_child_at_position(self, group, position):
        children = [None]
        child = self._groups[group].get_first_child()
        while child is not None:
            children.append(child)
            child = child.get_next_sibling()
        return children[position]

    def _add_button(self, button, group, position):
        if group not in self._groups:
            if self._groups:
                self._add_separator()
            group_box = Gtk.Box()
            self._tool_box.append(group_box)
            self._groups[group] = group_box
        self._groups[group].insert_child_after(button, self._find_child_at_position(group, position))

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
        self._tool_box.append(sep)

    def set_message(self, s):
        self._message.set_label(s)