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
class ConfigureSubplotsTk(backend_tools.ConfigureSubplotsBase):

    def trigger(self, *args):
        NavigationToolbar2Tk.configure_subplots(self)