import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
def _get_running_interactive_framework():
    """
    Return the interactive framework whose event loop is currently running, if
    any, or "headless" if no event loop can be started, or None.

    Returns
    -------
    Optional[str]
        One of the following values: "qt", "gtk3", "gtk4", "wx", "tk",
        "macosx", "headless", ``None``.
    """
    QtWidgets = sys.modules.get('PyQt6.QtWidgets') or sys.modules.get('PySide6.QtWidgets') or sys.modules.get('PyQt5.QtWidgets') or sys.modules.get('PySide2.QtWidgets')
    if QtWidgets and QtWidgets.QApplication.instance():
        return 'qt'
    Gtk = sys.modules.get('gi.repository.Gtk')
    if Gtk:
        if Gtk.MAJOR_VERSION == 4:
            from gi.repository import GLib
            if GLib.main_depth():
                return 'gtk4'
        if Gtk.MAJOR_VERSION == 3 and Gtk.main_level():
            return 'gtk3'
    wx = sys.modules.get('wx')
    if wx and wx.GetApp():
        return 'wx'
    tkinter = sys.modules.get('tkinter')
    if tkinter:
        codes = {tkinter.mainloop.__code__, tkinter.Misc.mainloop.__code__}
        for frame in sys._current_frames().values():
            while frame:
                if frame.f_code in codes:
                    return 'tk'
                frame = frame.f_back
        del frame
    macosx = sys.modules.get('matplotlib.backends._macosx')
    if macosx and macosx.event_loop_is_running():
        return 'macosx'
    if not _c_internal_utils.display_is_valid():
        return 'headless'
    return None