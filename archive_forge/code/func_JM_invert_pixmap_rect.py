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
def JM_invert_pixmap_rect(dest, b):
    """
    invert a rectangle - also supports non-alpha pixmaps
    """
    assert isinstance(dest, mupdf.FzPixmap)
    assert isinstance(b, mupdf.FzIrect)
    b = mupdf.fz_intersect_irect(b, mupdf.fz_pixmap_bbox(dest))
    w = b.x1 - b.x0
    y = b.y1 - b.y0
    if w <= 0 or y <= 0:
        return 0
    destspan = dest.stride()
    destp = destspan * (b.y0 - dest.y()) + dest.n() * (b.x0 - dest.x())
    n0 = dest.n() - dest.alpha()
    alpha = dest.alpha()
    while 1:
        s = destp
        for x in range(w):
            for i in range(n0):
                ss = mupdf.fz_samples_get(dest, s)
                ss = 255 - ss
                mupdf.fz_samples_set(dest, s, ss)
                s += 1
            if alpha:
                ss = mupdf.fz_samples_get(dest, s)
                ss += 1
                mupdf.fz_samples_set(dest, s, ss)
        destp += destspan
        y -= 1
        if y == 0:
            break
    return 1