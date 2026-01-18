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
class RadioButton(Widget):

    def __init__(self, label, i, var, callback):
        self.creator = partial(tk.Radiobutton, text=label, var=var, value=i, command=callback)