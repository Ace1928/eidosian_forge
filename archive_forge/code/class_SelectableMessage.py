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
class SelectableMessage(ttk.Frame):
    """
    A disabled Text widget which allows selection of text.  On the mac
    the selection does not highlight correctly unless the Text widget has
    focus and does not clear correctly unless the state is NORMAL.
    """

    def __init__(self, container):
        self.container = container
        ttk.Frame.__init__(self, container)
        self.scrollbar = AutoScrollbar(self, orient=Tk_.VERTICAL)
        self.text = text = Tk_.Text(self, width=60, height=12, highlightthickness=0, relief=Tk_.FLAT, yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=text.yview)
        self.grid_columnconfigure(0, weight=1)
        text.grid(row=0, column=0, sticky=Tk_.NSEW)
        text.bind('<<Copy>>', self.copy)
        text.bind('<Button-1>', lambda *args: self.text.focus_set())
        text.bind('<FocusOut>', self.focus_out)
        self.scrollbar.grid(row=0, column=1, sticky=Tk_.NS)
        self.text.config(state=Tk_.DISABLED)

    def focus_out(self, *args):
        self.text.config(state=Tk_.NORMAL)
        self.text.selection_clear()
        self.text.config(state=Tk_.DISABLED)

    def set(self, message):
        self.text.config(state=Tk_.NORMAL)
        self.text.delete('0.1', Tk_.END)
        self.text.selection_clear()
        self.text.insert(Tk_.INSERT, message)
        self.text.config(state=Tk_.DISABLED)

    def get(self):
        return self.text.get('0.1', Tk_.END)

    def copy(self, event):
        self.text.selection_get(selection='CLIPBOARD')