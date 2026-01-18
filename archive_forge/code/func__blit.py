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
def _blit(argsid):
    """
    Thin wrapper to blit called via tkapp.call.

    *argsid* is a unique string identifier to fetch the correct arguments from
    the ``_blit_args`` dict, since arguments cannot be passed directly.
    """
    photoimage, dataptr, offsets, bboxptr, comp_rule = _blit_args.pop(argsid)
    if not photoimage.tk.call('info', 'commands', photoimage):
        return
    _tkagg.blit(photoimage.tk.interpaddr(), str(photoimage), dataptr, comp_rule, offsets, bboxptr)