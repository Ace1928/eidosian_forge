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
@backend_tools._register_tool_class(FigureCanvasTk)
class RubberbandTk(backend_tools.RubberbandBase):

    def draw_rubberband(self, x0, y0, x1, y1):
        NavigationToolbar2Tk.draw_rubberband(self._make_classic_style_pseudo_toolbar(), None, x0, y0, x1, y1)

    def remove_rubberband(self):
        NavigationToolbar2Tk.remove_rubberband(self._make_classic_style_pseudo_toolbar())