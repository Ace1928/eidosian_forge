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
def _newPage(self, pno=-1, width=595, height=842):
    """Make a new PDF page."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    if g_use_extra:
        extra._newPage(self.this, pno, width, height)
    else:
        pdf = _as_pdf_document(self)
        assert isinstance(pdf, mupdf.PdfDocument)
        mediabox = mupdf.FzRect(mupdf.FzRect.Fixed_UNIT)
        mediabox.x1 = width
        mediabox.y1 = height
        contents = mupdf.FzBuffer()
        if pno < -1:
            raise ValueError(MSG_BAD_PAGENO)
        resources = mupdf.pdf_add_new_dict(pdf, 1)
        page_obj = mupdf.pdf_add_page(pdf, mediabox, 0, resources, contents)
        mupdf.pdf_insert_page(pdf, pno, page_obj)
    self._reset_page_refs()
    return self[pno]