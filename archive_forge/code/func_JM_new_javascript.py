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
def JM_new_javascript(pdf, value):
    """
    make new PDF action object from JavaScript source
    Parameters are a PDF document and a Python string.
    Returns a PDF action object.
    """
    if value is None:
        return
    data = JM_StrAsChar(value)
    if data is None:
        return
    res = mupdf.fz_new_buffer_from_copied_data(data.encode('utf8'))
    source = mupdf.pdf_add_stream(pdf, res, mupdf.PdfObj(), 0)
    newaction = mupdf.pdf_add_new_dict(pdf, 4)
    mupdf.pdf_dict_put(newaction, PDF_NAME('S'), mupdf.pdf_new_name('JavaScript'))
    mupdf.pdf_dict_put(newaction, PDF_NAME('JS'), source)
    return newaction