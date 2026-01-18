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
def _get_redact_values(self):
    annot = self.this
    if mupdf.pdf_annot_type(annot) != mupdf.PDF_ANNOT_REDACT:
        return
    values = dict()
    try:
        obj = mupdf.pdf_dict_gets(mupdf.pdf_annot_obj(annot), 'RO')
        if obj.m_internal:
            JM_Warning("Ignoring redaction key '/RO'.")
            xref = mupdf.pdf_to_num(obj)
            values[dictkey_xref] = xref
        obj = mupdf.pdf_dict_gets(mupdf.pdf_annot_obj(annot), 'OverlayText')
        if obj.m_internal:
            text = mupdf.pdf_to_text_string(obj)
            values[dictkey_text] = JM_UnicodeFromStr(text)
        else:
            values[dictkey_text] = ''
        obj = mupdf.pdf_dict_get(mupdf.pdf_annot_obj(annot), PDF_NAME('Q'))
        align = 0
        if obj.m_internal:
            align = mupdf.pdf_to_int(obj)
        values[dictkey_align] = align
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        return
    val = values
    if not val:
        return val
    val['rect'] = self.rect
    text_color, fontname, fontsize = TOOLS._parse_da(self)
    val['text_color'] = text_color
    val['fontname'] = fontname
    val['fontsize'] = fontsize
    fill = self.colors['fill']
    val['fill'] = fill
    return val