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
def _update_window_dpi(self, *args):
    newdpi = self._window_dpi.get()
    self.window.call('tk', 'scaling', newdpi / 72)
    if self.toolbar and hasattr(self.toolbar, '_rescale'):
        self.toolbar._rescale()
    self.canvas._update_device_pixel_ratio()