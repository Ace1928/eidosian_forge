import re
import sys
import tkinter
import tkinter.ttk as ttk
from warnings import warn
from .std import TqdmExperimentalWarning, TqdmWarning
from .std import tqdm as std_tqdm
@staticmethod
def _tk_dispatching_helper():
    """determine if Tkinter mainloop is dispatching events"""
    codes = {tkinter.mainloop.__code__, tkinter.Misc.mainloop.__code__}
    for frame in sys._current_frames().values():
        while frame:
            if frame.f_code in codes:
                return True
            frame = frame.f_back
    return False