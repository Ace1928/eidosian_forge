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
def JM_get_fontextension(doc, xref):
    """
    Return the file extension of a font file, identified by xref
    """
    if xref < 1:
        return 'n/a'
    o = mupdf.pdf_load_object(doc, xref)
    desft = mupdf.pdf_dict_get(o, PDF_NAME('DescendantFonts'))
    if desft.m_internal:
        obj = mupdf.pdf_resolve_indirect(mupdf.pdf_array_get(desft, 0))
        obj = mupdf.pdf_dict_get(obj, PDF_NAME('FontDescriptor'))
    else:
        obj = mupdf.pdf_dict_get(o, PDF_NAME('FontDescriptor'))
    if not obj.m_internal:
        return 'n/a'
    o = obj
    obj = mupdf.pdf_dict_get(o, PDF_NAME('FontFile'))
    if obj.m_internal:
        return 'pfa'
    obj = mupdf.pdf_dict_get(o, PDF_NAME('FontFile2'))
    if obj.m_internal:
        return 'ttf'
    obj = mupdf.pdf_dict_get(o, PDF_NAME('FontFile3'))
    if obj.m_internal:
        obj = mupdf.pdf_dict_get(obj, PDF_NAME('Subtype'))
        if obj.m_internal and (not mupdf.pdf_is_name(obj)):
            PySys_WriteStdout('invalid font descriptor subtype')
            return 'n/a'
        if mupdf.pdf_name_eq(obj, PDF_NAME('Type1C')):
            return 'cff'
        elif mupdf.pdf_name_eq(obj, PDF_NAME('CIDFontType0C')):
            return 'cid'
        elif mupdf.pdf_name_eq(obj, PDF_NAME('OpenType')):
            return 'otf'
        else:
            PySys_WriteStdout("unhandled font type '%s'", mupdf.pdf_to_name(obj))
    return 'n/a'