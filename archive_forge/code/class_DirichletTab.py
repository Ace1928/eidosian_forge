import sys
import os
from .gui import *
from .polyviewer import PolyhedronViewer
from .horoviewer import HoroballViewer
from .CyOpenGL import GetColor
from .app_menus import browser_menus
from . import app_menus
from .number import Number
from . import database
from .exceptions import SnapPeaFatalError
from plink import LinkViewer, LinkEditor
from plink.ipython_tools import IPythonTkRoot
from spherogram.links.orthogonal import OrthogonalLinkDiagram
class DirichletTab(PolyhedronViewer):

    def __init__(self, container, facedicts=[], title='Polyhedron Tab', parent=None):
        self.main_window = main_window
        self.style = style = SnapPyStyle()
        PolyhedronViewer.__init__(self, container, facedicts=facedicts, title=title, bgcolor=style.groupBG)

    def update_menus(self, menubar):
        menubar.children['help'].activate([app_menus.help_polyhedron_viewer_label, app_menus.help_report_bugs_label])