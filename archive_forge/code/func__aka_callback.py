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
def _aka_callback(self):
    self.after_cancel(self.aka_after_id)
    self.aka_after_id = None
    if self.identifier.state == 'finished':
        self._write_aka_info(self.identifier.get())
    else:
        self.aka_after_id = self.after(1000, self._aka_callback)