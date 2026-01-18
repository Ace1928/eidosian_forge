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
def JM_quad_from_py(r):
    if isinstance(r, mupdf.FzQuad):
        return r
    if hasattr(r, '__getitem__') and len(r) == 4 and hasattr(r[0], '__float__'):
        r = mupdf.FzRect(*tuple(r))
    if isinstance(r, mupdf.FzRect):
        return mupdf.fz_quad_from_rect(r)
    if isinstance(r, Quad):
        return mupdf.fz_make_quad(r.ul.x, r.ul.y, r.ur.x, r.ur.y, r.ll.x, r.ll.y, r.lr.x, r.lr.y)
    q = mupdf.fz_make_quad(0, 0, 0, 0, 0, 0, 0, 0)
    p = [0, 0, 0, 0]
    if not r or not isinstance(r, (tuple, list)) or len(r) != 4:
        return q
    if JM_FLOAT_ITEM(r, 0) is None:
        return mupdf.fz_quad_from_rect(JM_rect_from_py(r))
    for i in range(4):
        if i >= len(r):
            return q
        obj = r[i]
        if not PySequence_Check(obj) or PySequence_Size(obj) != 2:
            return q
        p[i].x = JM_FLOAT_ITEM(obj, 0)
        p[i].y = JM_FLOAT_ITEM(obj, 1)
        if p[i].x is None or p[i].y is None:
            return q
        p[i].x = max(p[i].x, FZ_MIN_INF_RECT)
        p[i].y = max(p[i].y, FZ_MIN_INF_RECT)
        p[i].x = min(p[i].x, FZ_MAX_INF_RECT)
        p[i].y = min(p[i].y, FZ_MAX_INF_RECT)
    q.ul = p[0]
    q.ur = p[1]
    q.ll = p[2]
    q.lr = p[3]
    return q