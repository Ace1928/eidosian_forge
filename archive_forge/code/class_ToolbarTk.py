import uuid
import weakref
from contextlib import contextmanager
import logging
import math
import os.path
import pathlib
import sys
import tkinter as tk
import tkinter.filedialog
import tkinter.font
import tkinter.messagebox
from tkinter.simpledialog import SimpleDialog
import numpy as np
from PIL import Image, ImageTk
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook, _c_internal_utils
from matplotlib.backend_bases import (
from matplotlib._pylab_helpers import Gcf
from . import _tkagg
class ToolbarTk(ToolContainerBase, tk.Frame):

    def __init__(self, toolmanager, window=None):
        ToolContainerBase.__init__(self, toolmanager)
        if window is None:
            window = self.toolmanager.canvas.get_tk_widget().master
        xmin, xmax = self.toolmanager.canvas.figure.bbox.intervalx
        height, width = (50, xmax - xmin)
        tk.Frame.__init__(self, master=window, width=int(width), height=int(height), borderwidth=2)
        self._label_font = tkinter.font.Font(size=10)
        label = tk.Label(master=self, font=self._label_font, text='\xa0\n\xa0')
        label.pack(side=tk.RIGHT)
        self._message = tk.StringVar(master=self)
        self._message_label = tk.Label(master=self, font=self._label_font, textvariable=self._message)
        self._message_label.pack(side=tk.RIGHT)
        self._toolitems = {}
        self.pack(side=tk.TOP, fill=tk.X)
        self._groups = {}

    def _rescale(self):
        return NavigationToolbar2Tk._rescale(self)

    def add_toolitem(self, name, group, position, image_file, description, toggle):
        frame = self._get_groupframe(group)
        buttons = frame.pack_slaves()
        if position >= len(buttons) or position < 0:
            before = None
        else:
            before = buttons[position]
        button = NavigationToolbar2Tk._Button(frame, name, image_file, toggle, lambda: self._button_click(name))
        button.pack_configure(before=before)
        if description is not None:
            ToolTip.createToolTip(button, description)
        self._toolitems.setdefault(name, [])
        self._toolitems[name].append(button)

    def _get_groupframe(self, group):
        if group not in self._groups:
            if self._groups:
                self._add_separator()
            frame = tk.Frame(master=self, borderwidth=0)
            frame.pack(side=tk.LEFT, fill=tk.Y)
            frame._label_font = self._label_font
            self._groups[group] = frame
        return self._groups[group]

    def _add_separator(self):
        return NavigationToolbar2Tk._Spacer(self)

    def _button_click(self, name):
        self.trigger_tool(name)

    def toggle_toolitem(self, name, toggled):
        if name not in self._toolitems:
            return
        for toolitem in self._toolitems[name]:
            if toggled:
                toolitem.select()
            else:
                toolitem.deselect()

    def remove_toolitem(self, name):
        for toolitem in self._toolitems[name]:
            toolitem.pack_forget()
        del self._toolitems[name]

    def set_message(self, s):
        self._message.set(s)