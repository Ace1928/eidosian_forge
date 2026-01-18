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
def _add_style_menu(self):
    style_menu = Tk_.Menu(self.menubar, tearoff=0)
    style_menu.add_radiobutton(label='PL', value='pl', command=self.set_style, variable=self.style_var)
    style_menu.add_radiobutton(label='Smooth', value='smooth', command=self.set_style, variable=self.style_var)
    self._extend_style_menu(style_menu)
    self.menubar.add_cascade(label='Style', menu=style_menu)
    self._add_zoom_and_pan(style_menu)