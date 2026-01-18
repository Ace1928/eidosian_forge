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
def _le_slash(annot, p1, p2, lr, fill_color):
    """Make stream commands for slash line end symbol. "lr" denotes left (False) or right point.
        """
    m, im, L, R, w, scol, fcol, opacity = TOOLS._le_annot_parms(annot, p1, p2, fill_color)
    rw = 1.1547 * max(1, w) * 1.0
    M = R if lr else L
    r = Rect(M.x - rw, M.y - 2 * w, M.x + rw, M.y + 2 * w)
    top = r.tl * im
    bot = r.br * im
    ap = '\nq\n%s%f %f m\n' % (opacity, top.x, top.y)
    ap += '%f %f l\n' % (bot.x, bot.y)
    ap += '%g w\n' % w
    ap += scol + 's\nQ\n'
    return ap