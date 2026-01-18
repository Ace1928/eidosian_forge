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
class TimerTk(TimerBase):
    """Subclass of `backend_bases.TimerBase` using Tk timer events."""

    def __init__(self, parent, *args, **kwargs):
        self._timer = None
        super().__init__(*args, **kwargs)
        self.parent = parent

    def _timer_start(self):
        self._timer_stop()
        self._timer = self.parent.after(self._interval, self._on_timer)

    def _timer_stop(self):
        if self._timer is not None:
            self.parent.after_cancel(self._timer)
        self._timer = None

    def _on_timer(self):
        super()._on_timer()
        if not self._single and self._timer:
            if self._interval > 0:
                self._timer = self.parent.after(self._interval, self._on_timer)
            else:
                self._timer = self.parent.after_idle(lambda: self.parent.after(self._interval, self._on_timer))
        else:
            self._timer = None