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
def JM_insert_font(pdf, bfname, fontfile, fontbuffer, set_simple, idx, wmode, serif, encoding, ordering):
    """
    Insert a font in a PDF
    """
    font = None
    res = None
    data = None
    ixref = 0
    index = 0
    simple = 0
    value = None
    name = None
    subt = None
    exto = None
    ENSURE_OPERATION(pdf)
    if ordering > -1:
        data, size, index = mupdf.fz_lookup_cjk_font(ordering)
    if data:
        font = mupdf.fz_new_font_from_memory(None, data, size, index, 0)
        font_obj = mupdf.pdf_add_cjk_font(pdf, font, ordering, wmode, serif)
        exto = 'n/a'
        simple = 0
    else:
        if bfname:
            data, size = mupdf.fz_lookup_base14_font(bfname)
        if data:
            font = mupdf.fz_new_font_from_memory(bfname, data, size, 0, 0)
            font_obj = mupdf.pdf_add_simple_font(pdf, font, encoding)
            exto = 'n/a'
            simple = 1
        else:
            if fontfile:
                font = mupdf.fz_new_font_from_file(None, fontfile, idx, 0)
            else:
                res = JM_BufferFromBytes(fontbuffer)
                if not res.m_internal:
                    RAISEPY(MSG_FILE_OR_BUFFER, PyExc_ValueError)
                font = mupdf.fz_new_font_from_buffer(None, res, idx, 0)
            if not set_simple:
                font_obj = mupdf.pdf_add_cid_font(pdf, font)
                simple = 0
            else:
                font_obj = mupdf.pdf_add_simple_font(pdf, font, encoding)
                simple = 2
    ixref = mupdf.pdf_to_num(font_obj)
    name = JM_EscapeStrFromStr(mupdf.pdf_to_name(mupdf.pdf_dict_get(font_obj, PDF_NAME('BaseFont'))))
    subt = JM_UnicodeFromStr(mupdf.pdf_to_name(mupdf.pdf_dict_get(font_obj, PDF_NAME('Subtype'))))
    if not exto:
        exto = JM_UnicodeFromStr(JM_get_fontextension(pdf, ixref))
    asc = mupdf.fz_font_ascender(font)
    dsc = mupdf.fz_font_descender(font)
    value = [ixref, {'name': name, 'type': subt, 'ext': exto, 'simple': bool(simple), 'ordering': ordering, 'ascender': asc, 'descender': dsc}]
    return value