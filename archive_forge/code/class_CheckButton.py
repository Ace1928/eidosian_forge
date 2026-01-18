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
class CheckButton(Widget):

    def __init__(self, text, value=False, callback=None):
        self.text = text
        self.var = tk.BooleanVar(value=value)
        self.callback = callback

    def create(self, parent):
        self.check = tk.Checkbutton(parent, text=self.text, var=self.var, command=self.callback)
        return self.check

    @property
    def value(self):
        return self.var.get()