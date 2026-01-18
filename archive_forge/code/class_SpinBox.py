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
class SpinBox(Widget):

    def __init__(self, value, start, end, step, callback=None, rounding=None, width=6):
        self.callback = callback
        self.rounding = rounding
        self.creator = partial(tk.Spinbox, from_=start, to=end, increment=step, command=callback, width=width)
        self.initial = str(value)

    def create(self, parent):
        self.widget = self.creator(parent)
        self.widget.bind('<Return>', lambda event: self.callback())
        self.value = self.initial
        return self.widget

    @property
    def value(self):
        x = self.widget.get().replace(',', '.')
        if '.' in x:
            return float(x)
        if x == 'None':
            return None
        return int(x)

    @value.setter
    def value(self, x):
        self.widget.delete(0, 'end')
        if '.' in str(x) and self.rounding is not None:
            try:
                x = round(float(x), self.rounding)
            except (ValueError, TypeError):
                pass
        self.widget.insert(0, x)