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
def JM_gather_fonts(pdf, dict_, fontlist, stream_xref):
    rc = 1
    n = mupdf.pdf_dict_len(dict_)
    for i in range(n):
        refname = mupdf.pdf_dict_get_key(dict_, i)
        fontdict = mupdf.pdf_dict_get_val(dict_, i)
        if not mupdf.pdf_is_dict(fontdict):
            mupdf.fz_warn(f"'{mupdf.pdf_to_name(refname)}' is no font dict ({mupdf.pdf_to_num(fontdict)} 0 R)")
            continue
        subtype = mupdf.pdf_dict_get(fontdict, mupdf.PDF_ENUM_NAME_Subtype)
        basefont = mupdf.pdf_dict_get(fontdict, mupdf.PDF_ENUM_NAME_BaseFont)
        if not basefont.m_internal or mupdf.pdf_is_null(basefont):
            name = mupdf.pdf_dict_get(fontdict, mupdf.PDF_ENUM_NAME_Name)
        else:
            name = basefont
        encoding = mupdf.pdf_dict_get(fontdict, mupdf.PDF_ENUM_NAME_Encoding)
        if mupdf.pdf_is_dict(encoding):
            encoding = mupdf.pdf_dict_get(encoding, mupdf.PDF_ENUM_NAME_BaseEncoding)
        xref = mupdf.pdf_to_num(fontdict)
        ext = 'n/a'
        if xref:
            ext = JM_get_fontextension(pdf, xref)
        entry = (xref, ext, mupdf.pdf_to_name(subtype), JM_EscapeStrFromStr(mupdf.pdf_to_name(name)), mupdf.pdf_to_name(refname), mupdf.pdf_to_name(encoding), stream_xref)
        fontlist.append(entry)
    return rc