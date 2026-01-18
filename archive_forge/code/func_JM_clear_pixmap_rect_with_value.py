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
def JM_clear_pixmap_rect_with_value(dest, value, b):
    """
    Clear a pixmap rectangle - my version also supports non-alpha pixmaps
    """
    b = mupdf.fz_intersect_irect(b, mupdf.fz_pixmap_bbox(dest))
    w = b.x1 - b.x0
    y = b.y1 - b.y0
    if w <= 0 or y <= 0:
        return 0
    destspan = dest.stride()
    destp = destspan * (b.y0 - dest.y()) + dest.n() * (b.x0 - dest.x())
    if mupdf.fz_colorspace_n(dest.colorspace()) == 4:
        value = 255 - value
        while 1:
            s = destp
            for x in range(0, w):
                mupdf.fz_samples_set(dest, s, 0)
                s += 1
                mupdf.fz_samples_set(dest, s, 0)
                s += 1
                mupdf.fz_samples_set(dest, s, 0)
                s += 1
                mupdf.fz_samples_set(dest, s, value)
                s += 1
                if dest.alpha():
                    mupdf.fz_samples_set(dest, s, 255)
                    s += 1
            destp += destspan
            if y == 0:
                break
            y -= 1
        return 1
    while 1:
        s = destp
        for x in range(w):
            for k in range(dest.n() - 1):
                mupdf.fz_samples_set(dest, s, value)
                s += 1
            if dest.alpha():
                mupdf.fz_samples_set(dest, s, 255)
                s += 1
            else:
                mupdf.fz_samples_set(dest, s, value)
                s += 1
        destp += destspan
        if y == 0:
            break
        y -= 1
    return 1