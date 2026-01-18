import re
import sys
from collections import namedtuple
from functools import partial
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.messagebox import askokcancel as ask_question
from tkinter.messagebox import showerror, showwarning, showinfo
from tkinter.filedialog import LoadFileDialog, SaveFileDialog
from ase.gui.i18n import _
def about(name, version, webpage):
    text = [name, '', _('Version') + ': ' + version, _('Web-page') + ': ' + webpage]
    win = Window(_('About'))
    win.add(Text('\n'.join(text)))