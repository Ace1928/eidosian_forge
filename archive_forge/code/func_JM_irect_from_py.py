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
def JM_irect_from_py(r):
    """
    PySequence to mupdf.FzIrect. Default: infinite irect
    """
    if isinstance(r, mupdf.FzIrect):
        return r
    if isinstance(r, IRect):
        r = mupdf.FzIrect(r.x0, r.y0, r.x1, r.y1)
        return r
    if isinstance(r, Rect):
        ret = mupdf.FzRect(r.x0, r.y0, r.x1, r.y1)
        ret = mupdf.FzIrect(ret)
        return ret
    if isinstance(r, mupdf.FzRect):
        ret = mupdf.FzIrect(r)
        return ret
    if not r or not PySequence_Check(r) or PySequence_Size(r) != 4:
        return mupdf.FzIrect(mupdf.fz_infinite_irect)
    f = [0, 0, 0, 0]
    for i in range(4):
        f[i] = r[i]
        if f[i] is None:
            return mupdf.FzIrect(mupdf.fz_infinite_irect)
        if f[i] < FZ_MIN_INF_RECT:
            f[i] = FZ_MIN_INF_RECT
        if f[i] > FZ_MAX_INF_RECT:
            f[i] = FZ_MAX_INF_RECT
    return mupdf.fz_make_irect(f[0], f[1], f[2], f[3])