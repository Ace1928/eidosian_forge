import os, time, webbrowser
from .gui import *
from . import smooth
from .vertex import Vertex
from .arrow import Arrow
from .crossings import Crossing, ECrossing
from .colors import Palette
from .dialog import InfoDialog
from .manager import LinkManager
from .viewer import LinkViewer
from .version import version
from .ipython_tools import IPythonTkRoot
class LinkDisplay(PLinkBase):
    """
    Displays an immutable link diagram.
    """

    def __init__(self, *args, **kwargs):
        if 'title' not in kwargs:
            kwargs['title'] = 'PLink Viewer'
        PLinkBase.__init__(self, *args, **kwargs)
        self.style_var.set('smooth')