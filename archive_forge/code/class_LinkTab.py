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
class LinkTab(LinkViewer):

    def __init__(self, data, window):
        self.style = style = SnapPyStyle()
        self.canvas = canvas = Tk_.Canvas(window, bg=style.groupBG, highlightthickness=0, highlightcolor=style.groupBG)
        LinkViewer.__init__(self, canvas, data)
        canvas.bind('<Configure>', lambda event: self.draw())

    def close(self, event=None):
        pass