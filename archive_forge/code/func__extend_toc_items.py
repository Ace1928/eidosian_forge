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
def _extend_toc_items(self, items):
    """Add color info to all items of an extended TOC list."""
    if self.is_closed:
        raise ValueError('document closed')
    if g_use_extra:
        return extra.Document_extend_toc_items(self.this, items)
    pdf = _as_pdf_document(self)
    zoom = 'zoom'
    bold = 'bold'
    italic = 'italic'
    collapse = 'collapse'
    root = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root'))
    if not root.m_internal:
        return
    olroot = mupdf.pdf_dict_get(root, PDF_NAME('Outlines'))
    if not olroot.m_internal:
        return
    first = mupdf.pdf_dict_get(olroot, PDF_NAME('First'))
    if not first.m_internal:
        return
    xrefs = []
    xrefs = JM_outline_xrefs(first, xrefs)
    n = len(xrefs)
    m = len(items)
    if not n:
        return
    if n != m:
        raise IndexError('internal error finding outline xrefs')
    for i in range(n):
        xref = int(xrefs[i])
        item = items[i]
        itemdict = item[3]
        if not isinstance(itemdict, dict):
            raise ValueError('need non-simple TOC format')
        itemdict[dictkey_xref] = xrefs[i]
        bm = mupdf.pdf_load_object(pdf, xref)
        flags = mupdf.pdf_to_int(mupdf.pdf_dict_get(bm, PDF_NAME('F')))
        if flags == 1:
            itemdict[italic] = True
        elif flags == 2:
            itemdict[bold] = True
        elif flags == 3:
            itemdict[italic] = True
            itemdict[bold] = True
        count = mupdf.pdf_to_int(mupdf.pdf_dict_get(bm, PDF_NAME('Count')))
        if count < 0:
            itemdict[collapse] = True
        elif count > 0:
            itemdict[collapse] = False
        col = mupdf.pdf_dict_get(bm, PDF_NAME('C'))
        if mupdf.pdf_is_array(col) and mupdf.pdf_array_len(col) == 3:
            color = (mupdf.pdf_to_real(mupdf.pdf_array_get(col, 0)), mupdf.pdf_to_real(mupdf.pdf_array_get(col, 1)), mupdf.pdf_to_real(mupdf.pdf_array_get(col, 2)))
            itemdict[dictkey_color] = color
        z = 0
        obj = mupdf.pdf_dict_get(bm, PDF_NAME('Dest'))
        if not obj.m_internal or not mupdf.pdf_is_array(obj):
            obj = mupdf.pdf_dict_getl(bm, PDF_NAME('A'), PDF_NAME('D'))
        if mupdf.pdf_is_array(obj) and mupdf.pdf_array_len(obj) == 5:
            z = mupdf.pdf_to_real(mupdf.pdf_array_get(obj, 4))
        itemdict[zoom] = float(z)
        item[3] = itemdict
        items[i] = item