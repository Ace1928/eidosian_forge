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
def _add_caret_annot(self, point):
    if g_use_extra:
        annot = extra._add_caret_annot(self.this, JM_point_from_py(point))
    elif g_use_extra:
        if isinstance(self.this, mupdf.PdfPage):
            page = self.this
        else:
            page = mupdf.pdf_page_from_fz_page(self.this)
        annot = extra._add_caret_annot(page, JM_point_from_py(point))
    else:
        page = self._pdf_page()
        annot = mupdf.pdf_create_annot(page, mupdf.PDF_ANNOT_CARET)
        if point:
            p = JM_point_from_py(point)
            r = mupdf.pdf_annot_rect(annot)
            r = mupdf.FzRect(p.x, p.y, p.x + r.x1 - r.x0, p.y + r.y1 - r.y0)
            mupdf.pdf_set_annot_rect(annot, r)
        mupdf.pdf_update_annot(annot)
        JM_add_annot_id(annot, 'A')
    return annot