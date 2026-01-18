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
def _get_char_widths(self, xref: int, bfname: str, ext: str, ordering: int, limit: int, idx: int=0):
    pdf = _as_pdf_document(self)
    mylimit = limit
    if mylimit < 256:
        mylimit = 256
    (ASSERT_PDF(pdf), f'pdf={pdf!r}')
    if ordering >= 0:
        data, size, index = mupdf.fz_lookup_cjk_font(ordering)
        font = mupdf.fz_new_font_from_memory(None, data, size, index, 0)
    else:
        data, size = mupdf.fz_lookup_base14_font(bfname)
        if data:
            font = mupdf.fz_new_font_from_memory(bfname, data, size, 0, 0)
        else:
            buf = JM_get_fontbuffer(pdf, xref)
            if not buf.m_internal:
                raise Exception('font at xref %d is not supported' % xref)
            font = mupdf.fz_new_font_from_buffer(None, buf, idx, 0)
    wlist = []
    for i in range(mylimit):
        glyph = mupdf.fz_encode_character(font, i)
        adv = mupdf.fz_advance_glyph(font, glyph, 0)
        if ordering >= 0:
            glyph = i
        if glyph > 0:
            wlist.append((glyph, adv))
        else:
            wlist.append((glyph, 0.0))
    return wlist