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
class DisplayList:

    def __del__(self):
        if not type(self) is DisplayList:
            return
        self.thisown = False

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], mupdf.FzRect):
            self.this = mupdf.FzDisplayList(args[0])
        elif len(args) == 1 and isinstance(args[0], mupdf.FzDisplayList):
            self.this = args[0]
        else:
            assert 0, f'Unrecognised args={args!r}'

    def get_pixmap(self, matrix=None, colorspace=None, alpha=0, clip=None):
        if isinstance(colorspace, Colorspace):
            colorspace = colorspace.this
        else:
            colorspace = mupdf.FzColorspace(mupdf.FzColorspace.Fixed_RGB)
        val = JM_pixmap_from_display_list(self.this, matrix, colorspace, alpha, clip, None)
        val.thisown = True
        return val

    def get_textpage(self, flags=3):
        stext_options = mupdf.FzStextOptions()
        stext_options.flags = flags
        val = mupdf.fz_new_stext_page_from_display_list(self.this, stext_options)
        val.thisown = True
        return val

    @property
    def rect(self):
        val = JM_py_from_rect(mupdf.fz_bound_display_list(self.this))
        val = Rect(val)
        return val

    def run(self, dw, m, area):
        mupdf.fz_run_display_list(self.this, dw.device, JM_matrix_from_py(m), JM_rect_from_py(area), mupdf.FzCookie())