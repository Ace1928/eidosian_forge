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
def _getPageInfo(self, pno, what):
    """List fonts, images, XObjects used on a page."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    doc = self.this
    pdf = _as_pdf_document(self)
    pageCount = mupdf.pdf_count_pages(doc) if isinstance(doc, mupdf.PdfDocument) else mupdf.fz_count_pages(doc)
    n = pno
    while n < 0:
        n += pageCount
    if n >= pageCount:
        raise ValueError(MSG_BAD_PAGENO)
    pageref = mupdf.pdf_lookup_page_obj(pdf, n)
    rsrc = mupdf.pdf_dict_get_inheritable(pageref, mupdf.PDF_ENUM_NAME_Resources)
    liste = []
    tracer = []
    if rsrc.m_internal:
        JM_scan_resources(pdf, rsrc, liste, what, 0, tracer)
    return liste