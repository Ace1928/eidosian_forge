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
def _embfile_add(self, name, buffer_, filename=None, ufilename=None, desc=None):
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    data = JM_BufferFromBytes(buffer_)
    if not data.m_internal:
        raise TypeError(MSG_BAD_BUFFER)
    names = mupdf.pdf_dict_getl(mupdf.pdf_trailer(pdf), PDF_NAME('Root'), PDF_NAME('Names'), PDF_NAME('EmbeddedFiles'), PDF_NAME('Names'))
    if not mupdf.pdf_is_array(names):
        root = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root'))
        names = mupdf.pdf_new_array(pdf, 6)
        mupdf.pdf_dict_putl(root, names, PDF_NAME('Names'), PDF_NAME('EmbeddedFiles'), PDF_NAME('Names'))
    fileentry = JM_embed_file(pdf, data, filename, ufilename, desc, 1)
    xref = mupdf.pdf_to_num(mupdf.pdf_dict_getl(fileentry, PDF_NAME('EF'), PDF_NAME('F')))
    mupdf.pdf_array_push(names, mupdf.pdf_new_text_string(name))
    mupdf.pdf_array_push(names, fileentry)
    return xref