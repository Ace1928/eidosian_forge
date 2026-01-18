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
def _add_line_annot(self, p1, p2):
    page = self._pdf_page()
    ASSERT_PDF(page)
    annot = mupdf.pdf_create_annot(page, mupdf.PDF_ANNOT_LINE)
    a = JM_point_from_py(p1)
    b = JM_point_from_py(p2)
    mupdf.pdf_set_annot_line(annot, a, b)
    mupdf.pdf_update_annot(annot)
    JM_add_annot_id(annot, 'A')
    assert annot.m_internal
    return Annot(annot)