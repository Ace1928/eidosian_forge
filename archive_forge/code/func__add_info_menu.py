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
def _add_info_menu(self):
    info_menu = Tk_.Menu(self.menubar, tearoff=0)
    info_menu.add_radiobutton(label='DT code', var=self.info_var, command=self.set_info, value=1)
    info_menu.add_radiobutton(label='Alphabetical DT', var=self.info_var, command=self.set_info, value=2)
    info_menu.add_radiobutton(label='Gauss code', var=self.info_var, command=self.set_info, value=3)
    info_menu.add_radiobutton(label='PD code', var=self.info_var, command=self.set_info, value=4)
    info_menu.add_radiobutton(label='BB framing', var=self.info_var, command=self.set_info, value=5)
    info_menu.add_separator()
    info_menu.add_checkbutton(label='DT labels', var=self.show_DT_var, command=self.update_info)
    if self.show_crossing_labels:
        info_menu.add_checkbutton(label='Crossing labels', var=self.show_labels_var, command=self.update_info)
    self.menubar.add_cascade(label='Info', menu=info_menu)