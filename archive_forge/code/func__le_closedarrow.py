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
def _le_closedarrow(annot, p1, p2, lr, fill_color):
    """Make stream commands for closed arrow line end symbol. "lr" denotes left (False) or right point.
        """
    m, im, L, R, w, scol, fcol, opacity = TOOLS._le_annot_parms(annot, p1, p2, fill_color)
    shift = 2.5
    d = shift * max(1, w)
    p2 = R + (d / 2.0, 0) if lr else L - (d / 2.0, 0)
    p1 = p2 + (-2 * d, -d) if lr else p2 + (2 * d, -d)
    p3 = p2 + (-2 * d, d) if lr else p2 + (2 * d, d)
    p1 *= im
    p2 *= im
    p3 *= im
    ap = '\nq\n%s%f %f m\n' % (opacity, p1.x, p1.y)
    ap += '%f %f l\n' % (p2.x, p2.y)
    ap += '%f %f l\n' % (p3.x, p3.y)
    ap += '%g w\n' % w
    ap += scol + fcol + 'b\nQ\n'
    return ap