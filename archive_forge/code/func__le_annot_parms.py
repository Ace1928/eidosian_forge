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
def _le_annot_parms(annot, p1, p2, fill_color):
    """Get common parameters for making annot line end symbols.

        Returns:
            m: matrix that maps p1, p2 to points L, P on the x-axis
            im: its inverse
            L, P: transformed p1, p2
            w: line width
            scol: stroke color string
            fcol: fill color store_shrink
            opacity: opacity string (gs command)
        """
    w = annot.border['width']
    sc = annot.colors['stroke']
    if not sc:
        sc = (0, 0, 0)
    scol = ' '.join(map(str, sc)) + ' RG\n'
    if fill_color:
        fc = fill_color
    else:
        fc = annot.colors['fill']
    if not fc:
        fc = (1, 1, 1)
    fcol = ' '.join(map(str, fc)) + ' rg\n'
    np1 = p1
    np2 = p2
    m = Matrix(util_hor_matrix(np1, np2))
    im = ~m
    L = np1 * m
    R = np2 * m
    if 0 <= annot.opacity < 1:
        opacity = '/H gs\n'
    else:
        opacity = ''
    return (m, im, L, R, w, scol, fcol, opacity)