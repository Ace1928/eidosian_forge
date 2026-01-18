import logging
import sys
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backend_tools import Cursors
import gi
from gi.repository import Gdk, Gio, GLib, Gtk
class _FigureManagerGTK(FigureManagerBase):
    """
    Attributes
    ----------
    canvas : `FigureCanvas`
        The FigureCanvas instance
    num : int or str
        The Figure number
    toolbar : Gtk.Toolbar or Gtk.Box
        The toolbar
    vbox : Gtk.VBox
        The Gtk.VBox containing the canvas and toolbar
    window : Gtk.Window
        The Gtk.Window
    """

    def __init__(self, canvas, num):
        self._gtk_ver = gtk_ver = Gtk.get_major_version()
        app = _create_application()
        self.window = Gtk.Window()
        app.add_window(self.window)
        super().__init__(canvas, num)
        if gtk_ver == 3:
            self.window.set_wmclass('matplotlib', 'Matplotlib')
            icon_ext = 'png' if sys.platform == 'win32' else 'svg'
            self.window.set_icon_from_file(str(cbook._get_data_path(f'images/matplotlib.{icon_ext}')))
        self.vbox = Gtk.Box()
        self.vbox.set_property('orientation', Gtk.Orientation.VERTICAL)
        if gtk_ver == 3:
            self.window.add(self.vbox)
            self.vbox.show()
            self.canvas.show()
            self.vbox.pack_start(self.canvas, True, True, 0)
        elif gtk_ver == 4:
            self.window.set_child(self.vbox)
            self.vbox.prepend(self.canvas)
        w, h = self.canvas.get_width_height()
        if self.toolbar is not None:
            if gtk_ver == 3:
                self.toolbar.show()
                self.vbox.pack_end(self.toolbar, False, False, 0)
            elif gtk_ver == 4:
                sw = Gtk.ScrolledWindow(vscrollbar_policy=Gtk.PolicyType.NEVER)
                sw.set_child(self.toolbar)
                self.vbox.append(sw)
            min_size, nat_size = self.toolbar.get_preferred_size()
            h += nat_size.height
        self.window.set_default_size(w, h)
        self._destroying = False
        self.window.connect('destroy', lambda *args: Gcf.destroy(self))
        self.window.connect({3: 'delete_event', 4: 'close-request'}[gtk_ver], lambda *args: Gcf.destroy(self))
        if mpl.is_interactive():
            self.window.show()
            self.canvas.draw_idle()
        self.canvas.grab_focus()

    def destroy(self, *args):
        if self._destroying:
            return
        self._destroying = True
        self.window.destroy()
        self.canvas.destroy()

    @classmethod
    def start_main_loop(cls):
        global _application
        if _application is None:
            return
        try:
            _application.run()
        except KeyboardInterrupt:
            context = GLib.MainContext.default()
            while context.pending():
                context.iteration(True)
            raise
        finally:
            _application = None

    def show(self):
        self.window.show()
        self.canvas.draw()
        if mpl.rcParams['figure.raise_window']:
            meth_name = {3: 'get_window', 4: 'get_surface'}[self._gtk_ver]
            if getattr(self.window, meth_name)():
                self.window.present()
            else:
                _api.warn_external('Cannot raise window yet to be setup')

    def full_screen_toggle(self):
        is_fullscreen = {3: lambda w: w.get_window().get_state() & Gdk.WindowState.FULLSCREEN, 4: lambda w: w.is_fullscreen()}[self._gtk_ver]
        if is_fullscreen(self.window):
            self.window.unfullscreen()
        else:
            self.window.fullscreen()

    def get_window_title(self):
        return self.window.get_title()

    def set_window_title(self, title):
        self.window.set_title(title)

    def resize(self, width, height):
        width = int(width / self.canvas.device_pixel_ratio)
        height = int(height / self.canvas.device_pixel_ratio)
        if self.toolbar:
            min_size, nat_size = self.toolbar.get_preferred_size()
            height += nat_size.height
        canvas_size = self.canvas.get_allocation()
        if self._gtk_ver >= 4 or canvas_size.width == canvas_size.height == 1:
            self.window.set_default_size(width, height)
        else:
            self.window.resize(width, height)