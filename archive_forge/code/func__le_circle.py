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
def _le_circle(annot, p1, p2, lr, fill_color):
    """Make stream commands for circle line end symbol. "lr" denotes left (False) or right point.
        """
    m, im, L, R, w, scol, fcol, opacity = TOOLS._le_annot_parms(annot, p1, p2, fill_color)
    shift = 2.5
    d = shift * max(1, w)
    M = R - (d / 2.0, 0) if lr else L + (d / 2.0, 0)
    r = Rect(M, M) + (-d, -d, d, d)
    ap = 'q\n' + opacity + TOOLS._oval_string(r.tl * im, r.tr * im, r.br * im, r.bl * im)
    ap += '%g w\n' % w
    ap += scol + fcol + 'b\nQ\n'
    return ap