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
def _delToC(self):
    """Delete the TOC."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    xrefs = []
    pdf = _as_pdf_document(self)
    if not pdf:
        return xrefs
    root = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root'))
    olroot = mupdf.pdf_dict_get(root, PDF_NAME('Outlines'))
    if not olroot.m_internal:
        return xrefs
    first = mupdf.pdf_dict_get(olroot, PDF_NAME('First'))
    xrefs = JM_outline_xrefs(first, xrefs)
    xref_count = len(xrefs)
    olroot_xref = mupdf.pdf_to_num(olroot)
    mupdf.pdf_delete_object(pdf, olroot_xref)
    mupdf.pdf_dict_del(root, PDF_NAME('Outlines'))
    for i in range(xref_count):
        _, xref = JM_INT_ITEM(xrefs, i)
        mupdf.pdf_delete_object(pdf, xref)
    xrefs.append(olroot_xref)
    val = xrefs
    self.init_doc()
    return val