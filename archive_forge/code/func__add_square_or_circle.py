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
def _add_square_or_circle(self, rect, annot_type):
    page = self._pdf_page()
    r = JM_rect_from_py(rect)
    if mupdf.fz_is_infinite_rect(r) or mupdf.fz_is_empty_rect(r):
        raise ValueError(MSG_BAD_RECT)
    annot = mupdf.pdf_create_annot(page, annot_type)
    mupdf.pdf_set_annot_rect(annot, r)
    mupdf.pdf_update_annot(annot)
    JM_add_annot_id(annot, 'A')
    assert annot.m_internal
    return Annot(annot)