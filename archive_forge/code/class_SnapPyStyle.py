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
class SnapPyStyle:

    def __init__(self):
        self.ttk_style = ttk_style = ttk.Style()
        if sys.platform == 'darwin':
            try:
                test = Tk_._default_root.winfo_rgb('systemWindowBackgroundColor1')
                self.windowBG = 'systemWindowBackgroundColor'
                self.groupBG = 'systemWindowBackgroundColor1'
                self.subgroupBG = 'systemWindowBackgroundColor2'
            except:
                self.windowBG = '#ededed'
                self.groupBG = '#e5e5e5'
                self.subgroupBG = '#dddddd'
        else:
            self.windowBG = ttk_style.lookup('TLabelframe', 'background')
            self.groupBG = self.subgroupBG = self.windowBG
        self.font = ttk_style.lookup('TLabel', 'font')
        self.font_info = fi = Font(font=self.font).actual()
        fi['size'] = abs(fi['size'])