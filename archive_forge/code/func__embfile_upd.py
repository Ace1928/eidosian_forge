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
def _embfile_upd(self, idx, buffer_=None, filename=None, ufilename=None, desc=None):
    pdf = _as_pdf_document(self)
    xref = 0
    names = mupdf.pdf_dict_getl(mupdf.pdf_trailer(pdf), PDF_NAME('Root'), PDF_NAME('Names'), PDF_NAME('EmbeddedFiles'), PDF_NAME('Names'))
    entry = mupdf.pdf_array_get(names, 2 * idx + 1)
    filespec = mupdf.pdf_dict_getl(entry, PDF_NAME('EF'), PDF_NAME('F'))
    if not filespec.m_internal:
        RAISEPY('bad PDF: no /EF object', JM_Exc_FileDataError)
    res = JM_BufferFromBytes(buffer_)
    if buffer_ and buffer_.m_internal and (not res.m_internal):
        raise TypeError(MSG_BAD_BUFFER)
    if res.m_internal and buffer_ and buffer_.m_internal:
        JM_update_stream(pdf, filespec, res, 1)
        len, _ = mupdf.fz_buffer_storage(res)
        l = mupdf.pdf_new_int(len)
        mupdf.pdf_dict_put(filespec, PDF_NAME('DL'), l)
        mupdf.pdf_dict_putl(filespec, l, PDF_NAME('Params'), PDF_NAME('Size'))
    xref = mupdf.pdf_to_num(filespec)
    if filename:
        mupdf.pdf_dict_put_text_string(entry, PDF_NAME('F'), filename)
    if ufilename:
        mupdf.pdf_dict_put_text_string(entry, PDF_NAME('UF'), ufilename)
    if desc:
        mupdf.pdf_dict_put_text_string(entry, PDF_NAME('Desc'), desc)
    return xref