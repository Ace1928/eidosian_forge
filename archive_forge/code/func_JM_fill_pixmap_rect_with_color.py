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
def JM_fill_pixmap_rect_with_color(dest, col, b):
    assert isinstance(dest, mupdf.FzPixmap)
    b = mupdf.fz_intersect_irect(b, mupdf.fz_pixmap_bbox(dest))
    w = b.x1 - b.x0
    y = b.y1 - b.y0
    if w <= 0 or y <= 0:
        return 0
    destspan = dest.stride()
    destp = destspan * (b.y0 - dest.y()) + dest.n() * (b.x0 - dest.x())
    while 1:
        s = destp
        for x in range(w):
            for i in range(dest.n()):
                mupdf.fz_samples_set(dest, s, col[i])
                s += 1
        destp += destspan
        y -= 1
        if y == 0:
            break
    return 1