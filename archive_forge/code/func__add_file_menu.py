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
def _add_file_menu(self):
    file_menu = Tk_.Menu(self.menubar, tearoff=0)
    file_menu.add_command(label='Open File ...', command=self.load)
    file_menu.add_command(label='Save ...', command=self.save)
    self.build_save_image_menu(self.menubar, file_menu)
    file_menu.add_separator()
    if self.callback:
        file_menu.add_command(label='Close', command=self.done)
    else:
        file_menu.add_command(label='Quit', command=self.done)
    self.menubar.add_cascade(label='File', menu=file_menu)