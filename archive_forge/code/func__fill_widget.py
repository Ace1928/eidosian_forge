import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
@staticmethod
def _fill_widget(annot, widget):
    val = JM_get_widget_properties(annot, widget)
    widget.rect = Rect(annot.rect)
    widget.xref = annot.xref
    widget.parent = annot.parent
    widget._annot = annot
    if not widget.script:
        widget.script = None
    if not widget.script_stroke:
        widget.script_stroke = None
    if not widget.script_format:
        widget.script_format = None
    if not widget.script_change:
        widget.script_change = None
    if not widget.script_calc:
        widget.script_calc = None
    if not widget.script_blur:
        widget.script_blur = None
    if not widget.script_focus:
        widget.script_focus = None
    return val