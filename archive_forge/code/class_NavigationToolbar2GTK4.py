import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
class NavigationToolbar2GTK4(_NavigationToolbar2GTK, Gtk.Box):

    def __init__(self, canvas):
        Gtk.Box.__init__(self)
        self.add_css_class('toolbar')
        self._gtk_ids = {}
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.append(Gtk.Separator())
                continue
            image = Gtk.Image.new_from_gicon(Gio.Icon.new_for_string(str(cbook._get_data_path('images', f'{image_file}-symbolic.svg'))))
            self._gtk_ids[text] = button = Gtk.ToggleButton() if callback in ['zoom', 'pan'] else Gtk.Button()
            button.set_child(image)
            button.add_css_class('flat')
            button.add_css_class('image-button')
            button._signal_handler = button.connect('clicked', getattr(self, callback))
            button.set_tooltip_text(tooltip_text)
            self.append(button)
        label = Gtk.Label()
        label.set_markup('<small>\xa0\n\xa0</small>')
        label.set_hexpand(True)
        self.append(label)
        self.message = Gtk.Label()
        self.message.set_justify(Gtk.Justification.RIGHT)
        self.append(self.message)
        _NavigationToolbar2GTK.__init__(self, canvas)

    def save_figure(self, *args):
        dialog = Gtk.FileChooserNative(title='Save the figure', transient_for=self.canvas.get_root(), action=Gtk.FileChooserAction.SAVE, modal=True)
        self._save_dialog = dialog
        ff = Gtk.FileFilter()
        ff.set_name('All files')
        ff.add_pattern('*')
        dialog.add_filter(ff)
        dialog.set_filter(ff)
        formats = []
        default_format = None
        for i, (name, fmts) in enumerate(self.canvas.get_supported_filetypes_grouped().items()):
            ff = Gtk.FileFilter()
            ff.set_name(name)
            for fmt in fmts:
                ff.add_pattern(f'*.{fmt}')
            dialog.add_filter(ff)
            formats.append(name)
            if self.canvas.get_default_filetype() in fmts:
                default_format = i
        formats = [formats[default_format], *formats[:default_format], *formats[default_format + 1:]]
        dialog.add_choice('format', 'File format', formats, formats)
        dialog.set_choice('format', formats[0])
        dialog.set_current_folder(Gio.File.new_for_path(os.path.expanduser(mpl.rcParams['savefig.directory'])))
        dialog.set_current_name(self.canvas.get_default_filename())

        @functools.partial(dialog.connect, 'response')
        def on_response(dialog, response):
            file = dialog.get_file()
            fmt = dialog.get_choice('format')
            fmt = self.canvas.get_supported_filetypes_grouped()[fmt][0]
            dialog.destroy()
            self._save_dialog = None
            if response != Gtk.ResponseType.ACCEPT:
                return
            if mpl.rcParams['savefig.directory']:
                parent = file.get_parent()
                mpl.rcParams['savefig.directory'] = parent.get_path()
            try:
                self.canvas.figure.savefig(file.get_path(), format=fmt)
            except Exception as e:
                msg = Gtk.MessageDialog(transient_for=self.canvas.get_root(), message_type=Gtk.MessageType.ERROR, buttons=Gtk.ButtonsType.OK, modal=True, text=str(e))
                msg.show()
        dialog.show()