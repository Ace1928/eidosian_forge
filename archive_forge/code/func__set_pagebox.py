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
def _set_pagebox(self, boxtype, rect):
    doc = self.parent
    if doc is None:
        raise ValueError('orphaned object: parent is None')
    if not doc.is_pdf:
        raise ValueError('is no PDF')
    valid_boxes = ('CropBox', 'BleedBox', 'TrimBox', 'ArtBox')
    if boxtype not in valid_boxes:
        raise ValueError('bad boxtype')
    rect = Rect(rect)
    mb = self.mediabox
    rect = Rect(rect[0], mb.y1 - rect[3], rect[2], mb.y1 - rect[1])
    if not (mb.x0 <= rect.x0 < rect.x1 <= mb.x1 and mb.y0 <= rect.y0 < rect.y1 <= mb.y1):
        raise ValueError(f'{boxtype} not in MediaBox')
    doc.xref_set_key(self.xref, boxtype, '[%g %g %g %g]' % tuple(rect))