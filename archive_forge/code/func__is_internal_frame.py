import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def _is_internal_frame(frame):
    """Signal whether the frame is a CPython or logging module internal."""
    filename = os.path.normcase(frame.f_code.co_filename)
    return filename == _srcfile or ('importlib' in filename and '_bootstrap' in filename)