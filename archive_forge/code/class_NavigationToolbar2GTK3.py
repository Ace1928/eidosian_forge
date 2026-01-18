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
class NavigationToolbar2GTK3(_NavigationToolbar2GTK, Gtk.Toolbar):

    def __init__(self, canvas):
        GObject.GObject.__init__(self)
        self.set_style(Gtk.ToolbarStyle.ICONS)
        self._gtk_ids = {}
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.insert(Gtk.SeparatorToolItem(), -1)
                continue
            image = Gtk.Image.new_from_gicon(Gio.Icon.new_for_string(str(cbook._get_data_path('images', f'{image_file}-symbolic.svg'))), Gtk.IconSize.LARGE_TOOLBAR)
            self._gtk_ids[text] = button = Gtk.ToggleToolButton() if callback in ['zoom', 'pan'] else Gtk.ToolButton()
            button.set_label(text)
            button.set_icon_widget(image)
            button._signal_handler = button.connect('clicked', getattr(self, callback))
            button.set_tooltip_text(tooltip_text)
            self.insert(button, -1)
        toolitem = Gtk.ToolItem()
        self.insert(toolitem, -1)
        label = Gtk.Label()
        label.set_markup('<small>\xa0\n\xa0</small>')
        toolitem.set_expand(True)
        toolitem.add(label)
        toolitem = Gtk.ToolItem()
        self.insert(toolitem, -1)
        self.message = Gtk.Label()
        self.message.set_justify(Gtk.Justification.RIGHT)
        toolitem.add(self.message)
        self.show_all()
        _NavigationToolbar2GTK.__init__(self, canvas)

    def save_figure(self, *args):
        dialog = Gtk.FileChooserDialog(title='Save the figure', parent=self.canvas.get_toplevel(), action=Gtk.FileChooserAction.SAVE, buttons=(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK))
        for name, fmts in self.canvas.get_supported_filetypes_grouped().items():
            ff = Gtk.FileFilter()
            ff.set_name(name)
            for fmt in fmts:
                ff.add_pattern(f'*.{fmt}')
            dialog.add_filter(ff)
            if self.canvas.get_default_filetype() in fmts:
                dialog.set_filter(ff)

        @functools.partial(dialog.connect, 'notify::filter')
        def on_notify_filter(*args):
            name = dialog.get_filter().get_name()
            fmt = self.canvas.get_supported_filetypes_grouped()[name][0]
            dialog.set_current_name(str(Path(dialog.get_current_name()).with_suffix(f'.{fmt}')))
        dialog.set_current_folder(mpl.rcParams['savefig.directory'])
        dialog.set_current_name(self.canvas.get_default_filename())
        dialog.set_do_overwrite_confirmation(True)
        response = dialog.run()
        fname = dialog.get_filename()
        ff = dialog.get_filter()
        fmt = self.canvas.get_supported_filetypes_grouped()[ff.get_name()][0]
        dialog.destroy()
        if response != Gtk.ResponseType.OK:
            return
        if mpl.rcParams['savefig.directory']:
            mpl.rcParams['savefig.directory'] = os.path.dirname(fname)
        try:
            self.canvas.figure.savefig(fname, format=fmt)
        except Exception as e:
            dialog = Gtk.MessageDialog(parent=self.canvas.get_toplevel(), message_format=str(e), type=Gtk.MessageType.ERROR, buttons=Gtk.ButtonsType.OK)
            dialog.run()
            dialog.destroy()