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
class RadioButtons(Widget):

    def __init__(self, labels, values=None, callback=None, vertical=False):
        self.var = tk.IntVar()
        if callback:

            def callback2():
                callback(self.value)
        else:
            callback2 = None
        self.values = values or list(range(len(labels)))
        self.buttons = [RadioButton(label, i, self.var, callback2) for i, label in enumerate(labels)]
        self.vertical = vertical

    def create(self, parent):
        self.widget = frame = tk.Frame(parent)
        side = 'top' if self.vertical else 'left'
        for button in self.buttons:
            button.create(frame).pack(side=side)
        return frame

    @property
    def value(self):
        return self.values[self.var.get()]

    @value.setter
    def value(self, value):
        self.var.set(self.values.index(value))

    def __getitem__(self, value):
        return self.buttons[self.values.index(value)]