import os
import sys
import re
import time
from collections.abc import Mapping  # Python 3.5 or newer
from IPython.core.displayhook import DisplayHook
from tkinter.messagebox import askyesno
from .gui import *
from . import filedialog
from .exceptions import SnapPeaFatalError
from .app_menus import HelpMenu, EditMenu, WindowMenu, ListedWindow
from .app_menus import dirichlet_menus, horoball_menus, inside_view_menus, plink_menus
from .app_menus import add_menu, scut, open_html_docs
from .browser import Browser
from .horoviewer import HoroballViewer
from .infowindow import about_snappy, InfoWindow
from .polyviewer import PolyhedronViewer
from .raytracing.inside_viewer import InsideViewer
from .settings import Settings, SettingsDialog
from .phone_home import update_needed
from .SnapPy import SnapPea_interrupt, msg_stream
from .shell import SnapPyInteractiveShellEmbed
from .tkterminal import TkTerm, snappy_path
from plink import LinkEditor
from plink.smooth import Smoother
import site
import pydoc
class SnapPyLinkEditor(LinkEditor, ListedWindow):

    def __init__(self, root=None, no_arcs=False, callback=None, cb_menu='', manifold=None, file_name=None):
        self.manifold = manifold
        self.main_window = terminal
        LinkEditor.__init__(self, root=terminal.window, no_arcs=no_arcs, callback=callback, cb_menu=cb_menu, manifold=manifold, file_name=file_name)
        self.set_title()
        self.register_window(self)
        self.window.focus_set()
        self.window.after_idle(self.set_title)

    def done(self, event=None):
        self.unregister_window(self)
        self.window.withdraw()

    def reopen(self):
        self.register_window(self)
        self.window.deiconify()

    def deiconify(self):
        self.window.deiconify()

    def lift(self):
        self.window.lift()

    def focus_force(self):
        self.window.focus_force()

    def set_title(self):
        title = 'Plink Editor'
        if self.IP:
            ns = self.IP.user_ns
            names = [name for name in ns if ns[name] is self.manifold]
            if names:
                names.sort(key=lambda x: '}' + x if x.startswith('_') else x)
                if names[0] == '_':
                    count = self.IP.execution_count
                    title += ' - Out[%d]' % count
                else:
                    title += ' - %s' % names[0]
            else:
                count = self.IP.execution_count
                if ns['_'] is self.manifold:
                    title += ' - Out[%d]' % count
        self.window.title(title)
        self.menu_title = title
    _build_menus = plink_menus

    def load(self, event=None, file_name=None):
        LinkEditor.load(self, file_name)

    def save(self, event=None):
        LinkEditor.save(self)

    def howto(self):
        open_html_docs('plink.html')
    __repr__ = object.__repr__