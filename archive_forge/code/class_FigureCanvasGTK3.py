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
class FigureCanvasGTK3(_FigureCanvasGTK, Gtk.DrawingArea):
    required_interactive_framework = 'gtk3'
    manager_class = _api.classproperty(lambda cls: FigureManagerGTK3)
    event_mask = Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK | Gdk.EventMask.EXPOSURE_MASK | Gdk.EventMask.KEY_PRESS_MASK | Gdk.EventMask.KEY_RELEASE_MASK | Gdk.EventMask.ENTER_NOTIFY_MASK | Gdk.EventMask.LEAVE_NOTIFY_MASK | Gdk.EventMask.POINTER_MOTION_MASK | Gdk.EventMask.SCROLL_MASK

    def __init__(self, figure=None):
        super().__init__(figure=figure)
        self._idle_draw_id = 0
        self._rubberband_rect = None
        self.connect('scroll_event', self.scroll_event)
        self.connect('button_press_event', self.button_press_event)
        self.connect('button_release_event', self.button_release_event)
        self.connect('configure_event', self.configure_event)
        self.connect('screen-changed', self._update_device_pixel_ratio)
        self.connect('notify::scale-factor', self._update_device_pixel_ratio)
        self.connect('draw', self.on_draw_event)
        self.connect('draw', self._post_draw)
        self.connect('key_press_event', self.key_press_event)
        self.connect('key_release_event', self.key_release_event)
        self.connect('motion_notify_event', self.motion_notify_event)
        self.connect('enter_notify_event', self.enter_notify_event)
        self.connect('leave_notify_event', self.leave_notify_event)
        self.connect('size_allocate', self.size_allocate)
        self.set_events(self.__class__.event_mask)
        self.set_can_focus(True)
        css = Gtk.CssProvider()
        css.load_from_data(b'.matplotlib-canvas { background-color: white; }')
        style_ctx = self.get_style_context()
        style_ctx.add_provider(css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        style_ctx.add_class('matplotlib-canvas')

    def destroy(self):
        CloseEvent('close_event', self)._process()

    def set_cursor(self, cursor):
        window = self.get_property('window')
        if window is not None:
            window.set_cursor(_mpl_to_gtk_cursor(cursor))
            context = GLib.MainContext.default()
            context.iteration(True)

    def _mpl_coords(self, event=None):
        """
        Convert the position of a GTK event, or of the current cursor position
        if *event* is None, to Matplotlib coordinates.

        GTK use logical pixels, but the figure is scaled to physical pixels for
        rendering.  Transform to physical pixels so that all of the down-stream
        transforms work as expected.

        Also, the origin is different and needs to be corrected.
        """
        if event is None:
            window = self.get_window()
            t, x, y, state = window.get_device_position(window.get_display().get_device_manager().get_client_pointer())
        else:
            x, y = (event.x, event.y)
        x = x * self.device_pixel_ratio
        y = self.figure.bbox.height - y * self.device_pixel_ratio
        return (x, y)

    def scroll_event(self, widget, event):
        step = 1 if event.direction == Gdk.ScrollDirection.UP else -1
        MouseEvent('scroll_event', self, *self._mpl_coords(event), step=step, modifiers=self._mpl_modifiers(event.state), guiEvent=event)._process()
        return False

    def button_press_event(self, widget, event):
        MouseEvent('button_press_event', self, *self._mpl_coords(event), event.button, modifiers=self._mpl_modifiers(event.state), guiEvent=event)._process()
        return False

    def button_release_event(self, widget, event):
        MouseEvent('button_release_event', self, *self._mpl_coords(event), event.button, modifiers=self._mpl_modifiers(event.state), guiEvent=event)._process()
        return False

    def key_press_event(self, widget, event):
        KeyEvent('key_press_event', self, self._get_key(event), *self._mpl_coords(), guiEvent=event)._process()
        return True

    def key_release_event(self, widget, event):
        KeyEvent('key_release_event', self, self._get_key(event), *self._mpl_coords(), guiEvent=event)._process()
        return True

    def motion_notify_event(self, widget, event):
        MouseEvent('motion_notify_event', self, *self._mpl_coords(event), modifiers=self._mpl_modifiers(event.state), guiEvent=event)._process()
        return False

    def enter_notify_event(self, widget, event):
        gtk_mods = Gdk.Keymap.get_for_display(self.get_display()).get_modifier_state()
        LocationEvent('figure_enter_event', self, *self._mpl_coords(event), modifiers=self._mpl_modifiers(gtk_mods), guiEvent=event)._process()

    def leave_notify_event(self, widget, event):
        gtk_mods = Gdk.Keymap.get_for_display(self.get_display()).get_modifier_state()
        LocationEvent('figure_leave_event', self, *self._mpl_coords(event), modifiers=self._mpl_modifiers(gtk_mods), guiEvent=event)._process()

    def size_allocate(self, widget, allocation):
        dpival = self.figure.dpi
        winch = allocation.width * self.device_pixel_ratio / dpival
        hinch = allocation.height * self.device_pixel_ratio / dpival
        self.figure.set_size_inches(winch, hinch, forward=False)
        ResizeEvent('resize_event', self)._process()
        self.draw_idle()

    @staticmethod
    def _mpl_modifiers(event_state, *, exclude=None):
        modifiers = [('ctrl', Gdk.ModifierType.CONTROL_MASK, 'control'), ('alt', Gdk.ModifierType.MOD1_MASK, 'alt'), ('shift', Gdk.ModifierType.SHIFT_MASK, 'shift'), ('super', Gdk.ModifierType.MOD4_MASK, 'super')]
        return [name for name, mask, key in modifiers if exclude != key and event_state & mask]

    def _get_key(self, event):
        unikey = chr(Gdk.keyval_to_unicode(event.keyval))
        key = cbook._unikey_or_keysym_to_mplkey(unikey, Gdk.keyval_name(event.keyval))
        mods = self._mpl_modifiers(event.state, exclude=key)
        if 'shift' in mods and unikey.isprintable():
            mods.remove('shift')
        return '+'.join([*mods, key])

    def _update_device_pixel_ratio(self, *args, **kwargs):
        if self._set_device_pixel_ratio(self.get_scale_factor()):
            self.queue_resize()
            self.queue_draw()

    def configure_event(self, widget, event):
        if widget.get_property('window') is None:
            return
        w = event.width * self.device_pixel_ratio
        h = event.height * self.device_pixel_ratio
        if w < 3 or h < 3:
            return
        dpi = self.figure.dpi
        self.figure.set_size_inches(w / dpi, h / dpi, forward=False)
        return False

    def _draw_rubberband(self, rect):
        self._rubberband_rect = rect
        self.queue_draw()

    def _post_draw(self, widget, ctx):
        if self._rubberband_rect is None:
            return
        x0, y0, w, h = (dim / self.device_pixel_ratio for dim in self._rubberband_rect)
        x1 = x0 + w
        y1 = y0 + h
        ctx.move_to(x0, y0)
        ctx.line_to(x0, y1)
        ctx.move_to(x0, y0)
        ctx.line_to(x1, y0)
        ctx.move_to(x0, y1)
        ctx.line_to(x1, y1)
        ctx.move_to(x1, y0)
        ctx.line_to(x1, y1)
        ctx.set_antialias(1)
        ctx.set_line_width(1)
        ctx.set_dash((3, 3), 0)
        ctx.set_source_rgb(0, 0, 0)
        ctx.stroke_preserve()
        ctx.set_dash((3, 3), 3)
        ctx.set_source_rgb(1, 1, 1)
        ctx.stroke()

    def on_draw_event(self, widget, ctx):
        pass

    def draw(self):
        if self.is_drawable():
            self.queue_draw()

    def draw_idle(self):
        if self._idle_draw_id != 0:
            return

        def idle_draw(*args):
            try:
                self.draw()
            finally:
                self._idle_draw_id = 0
            return False
        self._idle_draw_id = GLib.idle_add(idle_draw)

    def flush_events(self):
        context = GLib.MainContext.default()
        while context.pending():
            context.iteration(True)