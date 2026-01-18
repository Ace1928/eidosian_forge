import os
import sys
import time
import tempfile
import tkinter as Tk_
from tkinter import ttk as ttk
from tkinter.font import Font, families as font_families
from tkinter.simpledialog import Dialog, SimpleDialog
from plink.ipython_tools import IPythonTkRoot
from . import filedialog
class Spinbox(ttk.Entry):

    def __init__(self, container=None, **kw):
        ttk.Entry.__init__(self, container, 'ttk::spinbox', **kw)

    def set(self, value):
        self.tk.call(self._w, 'set', value)