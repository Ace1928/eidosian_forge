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
def _get_textpage(self, clip=None, flags=0, matrix=None):
    if g_use_extra:
        ll_tpage = extra.page_get_textpage(self.this, clip, flags, matrix)
        tpage = mupdf.FzStextPage(ll_tpage)
        return tpage
    page = self.this
    options = mupdf.FzStextOptions(flags)
    rect = JM_rect_from_py(clip)
    rect = mupdf.fz_bound_page(page) if clip is None else JM_rect_from_py(clip)
    ctm = JM_matrix_from_py(matrix)
    tpage = mupdf.FzStextPage(rect)
    dev = mupdf.fz_new_stext_device(tpage, options)
    if g_no_device_caching:
        mupdf.fz_enable_device_hints(dev, mupdf.FZ_NO_CACHE)
    if isinstance(page, mupdf.FzPage):
        pass
    elif isinstance(page, mupdf.PdfPage):
        page = page.super()
    else:
        assert 0, f'Unrecognised type(page)={type(page)!r}'
    mupdf.fz_run_page(page, dev, ctm, mupdf.FzCookie())
    mupdf.fz_close_device(dev)
    return tpage