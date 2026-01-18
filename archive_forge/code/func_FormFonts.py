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
@property
def FormFonts(self):
    """Get list of field font resource names."""
    pdf = _as_pdf_document(self)
    if not pdf:
        return
    fonts = mupdf.pdf_dict_getl(mupdf.pdf_trailer(pdf), PDF_NAME('Root'), PDF_NAME('AcroForm'), PDF_NAME('DR'), PDF_NAME('Font'))
    liste = list()
    if fonts.m_internal and mupdf.pdf_is_dict(fonts):
        n = mupdf.pdf_dict_len(fonts)
        for i in range(n):
            f = mupdf.pdf_dict_get_key(fonts, i)
            liste.append(JM_UnicodeFromStr(mupdf.pdf_to_name(f)))
    return liste