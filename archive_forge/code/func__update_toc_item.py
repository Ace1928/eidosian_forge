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
def _update_toc_item(self, xref, action=None, title=None, flags=0, collapse=None, color=None):
    """
        "update" bookmark by letting it point to nowhere
        """
    pdf = _as_pdf_document(self)
    item = mupdf.pdf_new_indirect(pdf, xref, 0)
    if title:
        mupdf.pdf_dict_put_text_string(item, PDF_NAME('Title'), title)
    if action:
        mupdf.pdf_dict_del(item, PDF_NAME('Dest'))
        obj = JM_pdf_obj_from_str(pdf, action)
        mupdf.pdf_dict_put(item, PDF_NAME('A'), obj)
    mupdf.pdf_dict_put_int(item, PDF_NAME('F'), flags)
    if color:
        c = mupdf.pdf_new_array(pdf, 3)
        for i in range(3):
            f = color[i]
            mupdf.pdf_array_push_real(c, f)
        mupdf.pdf_dict_put(item, PDF_NAME('C'), c)
    elif color is not None:
        mupdf.pdf_dict_del(item, PDF_NAME('C'))
    if collapse is not None:
        if mupdf.pdf_dict_get(item, PDF_NAME('Count')).m_internal:
            i = mupdf.pdf_dict_get_int(item, PDF_NAME('Count'))
            if i < 0 and collapse is False or (i > 0 and collapse is True):
                i = i * -1
                mupdf.pdf_dict_put_int(item, PDF_NAME('Count'), i)