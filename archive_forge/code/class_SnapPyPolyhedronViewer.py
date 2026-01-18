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
class SnapPyPolyhedronViewer(PolyhedronViewer):
    build_menus = dirichlet_menus

    def __init__(self, *args, **kwargs):
        PolyhedronViewer.__init__(self, *args, **kwargs, main_window=terminal)
        self.help_button.configure(command=self.help_window)

    def help_window(self):
        window = self.parent
        if not hasattr(window, 'polyhedron_help'):
            window.polyhedron_help = InfoWindow(window, 'Polyhedron Viewer Help', self.widget.help_text, 'polyhedron_help')
        else:
            window.polyhedron_help.deiconify()
            window.polyhedron_help.lift()
            window.polyhedron_help.focus_force()